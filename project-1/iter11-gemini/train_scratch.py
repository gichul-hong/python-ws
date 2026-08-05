# Iter10 (gemini): 3->5 혼동 방지(해상도 128) 및 소수 클래스 과예측 방지(sqrt weight)
#
# 목적: 지금까지 신뢰 가능한 실측값은 iter6의 test Macro F1 0.9892 단 하나뿐.
#       개선 여부를 판단할 수 있는 지표를 먼저 확보한다.
#
# iter6(0.9892) 대비 변경점은 아키텍처 1개로 제한:
#   - ResNet-50@96  ->  ConvNeXt-Tiny@96  (iter8에서 solo TTA 0.9964로 최강 확인)
#   - 증강은 iter6과 동일하게 롤백 (iter8의 blur 강화/저해상도 시뮬레이션 제거)
#
# iter8 실패 원인(확정):
#   원본 중앙값 42px에서 삼각형 내부 픽토그램은 약 20x20px.
#   96px 스케일의 GaussianBlur(k=5, sigma<=2.0)는 원본 기준 약 2.2px 블러로
#   픽토그램 선폭의 ~10%를 소실시켜 class 11 <-> 30 변별 단서를 파괴함.
#   또한 원본이 이미 25~225px(9배)를 자연 포함하므로 합성 저해상도는 불필요.
#
# 측정 설계:
#   - 트랙 단위 3-fold (클래스별 트랙을 round-robin 배분)
#     -> 소수 클래스(6트랙)도 fold당 2트랙 확보, 전 트랙이 정확히 1회 평가됨
#   - OOF(out-of-fold) 예측을 26,010장 전체에 대해 수집
#     -> 전체 데이터 기준 혼동 행렬 / 클래스별 F1 산출 (단일 20% split보다 신뢰도 높음)
#   - fold 평균으로 best-epoch 선택 편향 완화
#   - 부수 산출: 3-fold 모델 앙상블로 test 예측 -> 분포 프록시 비교(iter6 78/12, iter8 88/22)
#
# 제약 준수: test 라벨/분포/이미지 학습 미사용. 분포 체크는 사후 진단 전용.
import os
import re
import numpy as np
import cv2
import pandas as pd
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = r'C:\hong\python-ws\project-1\dataset\data 2'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_RESULT = r'C:\hong\python-ws\project-1\result.csv'
IMG_SIZE = 48            # Scratch CNN을 위한 48px 해상도
NUM_CLASSES = 43
N_FOLDS = 3
EPOCHS = 20              # iter8 ConvNeXt는 ep5에 0.9949, ep23에 0.9969 -> 20으로 충분
WARMUP_EPOCHS = 2
BATCH_SIZE = 64
LR_BACKBONE = 1e-3
LR_HEAD = 1e-3
SEED = 42
CLASS_WEIGHT_MODE = 'sqrt'       # 'balanced'(iter6과 동일) | 'sqrt'(완화) | 'none' - 소수 클래스 과예측 완화
SMALL_CLASSES = {0, 6, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 34, 36, 37, 39, 40, 41, 42}
# 집중 점검 대상 (iter8에서 관측된 혼동 쌍)
WATCH_PAIRS = [(11, 30), (30, 11), (26, 12), (12, 26)]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
print(f'Device: {DEVICE}')
print(f'Config: ConvNeXt-Tiny@{IMG_SIZE}, {N_FOLDS}-fold track CV, {EPOCHS}ep, '
      f'class_weight={CLASS_WEIGHT_MODE}')

# ---------------- Data loading (track ID 보존) ----------------
print('\n[1] Loading training data with track IDs...')
FNAME_RE = re.compile(r'^(\d{5})_(\d{5})_(\d{5})\.png$')
data, labels, tracks = [], [], []
train_path = os.path.join(DATA_ROOT, 'Train')
for i in range(NUM_CLASSES):
    class_dir = os.path.join(train_path, str(i))
    for fname in os.listdir(class_dir):
        m = FNAME_RE.match(fname)
        if not m:
            if fname.endswith('.png'):
                print(f'  [WARN] track parse failed: {fname} (class {i})')
            continue
        img = cv2.imread(os.path.join(class_dir, fname))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
        labels.append(i)
        tracks.append((i, int(m.group(2))))

data = np.array(data, dtype=np.uint8)
labels = np.array(labels, dtype=np.int64)
tracks = np.array(tracks, dtype=np.int64)
track_keys = [tuple(t) for t in tracks]
uniq_tracks = sorted(set(track_keys))
print(f'  Total: {len(data)} images, {len(uniq_tracks)} unique tracks')

# ---------------- Track-level 3-fold 배분 (클래스별 round-robin) ----------------
# round-robin: 소수 클래스(6트랙)도 fold당 정확히 2트랙 확보
print(f'\n[2] Track-level {N_FOLDS}-fold assignment (round-robin per class)...')
rng = np.random.RandomState(SEED)
class_tracks = defaultdict(list)
for t in uniq_tracks:
    class_tracks[t[0]].append(t)

track_fold = {}
for c in range(NUM_CLASSES):
    ts = class_tracks[c]
    order = rng.permutation(len(ts))          # 클래스 내 트랙 순서 무작위화
    for rank, idx in enumerate(order):
        track_fold[ts[idx]] = rank % N_FOLDS   # round-robin
fold_of_sample = np.array([track_fold[k] for k in track_keys], dtype=np.int64)

for f in range(N_FOLDS):
    n_img = int((fold_of_sample == f).sum())
    n_trk = sum(1 for v in track_fold.values() if v == f)
    # 소수 클래스별 val 트랙 수 확인 (측정 신뢰도의 핵심)
    small_trk = [sum(1 for t in class_tracks[c] if track_fold[t] == f) for c in sorted(SMALL_CLASSES)]
    print(f'  fold {f}: {n_img:>6} imgs, {n_trk:>3} tracks, '
          f'small-class val tracks min={min(small_trk)} max={max(small_trk)}')

# ---------------- Transforms: iter6과 동일 (blur/저해상도 제거) ----------------
normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
# 다수 클래스: 기하 + 광도만
base_tf = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    normalize,
])
# 소수 클래스: 기하/광도 강화. iter6에 있던 blur(k=3,sigma<=1.0,p=0.3)는
# 픽토그램 보존을 위해 제거, RandomErasing도 소폭 축소
small_tf = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.RandomAffine(degrees=18, translate=(0.15, 0.15), scale=(0.75, 1.25), shear=8),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    normalize,
    T.RandomErasing(p=0.2, scale=(0.02, 0.08)),
])
eval_tf = T.Compose([T.ToDtype(torch.float32, scale=True), normalize])


class TSDataset(Dataset):
    def __init__(self, images, labels, train=False):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.from_numpy(labels)
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x, y = self.images[idx], self.labels[idx]
        if self.train:
            tf = small_tf if int(y) in SMALL_CLASSES else base_tf
        else:
            tf = eval_tf
        return tf(x), y


class ScratchCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.10),
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.20),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        gap = x.mean([-2, -1])
        gmp = x.amax([-2, -1])
        x = torch.cat([gap, gmp], dim=1)
        return self.classifier(x)

def build_model():
    m = ScratchCNN(NUM_CLASSES)
    return m.to(DEVICE)


TTA_OPS = [
    lambda b: b,
    lambda b: TF.rotate(b, 8),
    lambda b: TF.rotate(b, -8),
    lambda b: TF.affine(b, angle=0, translate=[2, 2], scale=1.1, shear=[0.0]),
]


def predict_probs(model, images_uint8, batch=256):
    t_all = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).contiguous()
    model.eval()
    total = None
    with torch.no_grad():
        for op in TTA_OPS:
            probs = []
            for i in range(0, len(t_all), batch):
                x = eval_tf(t_all[i:i+batch])
                x = op(x).to(DEVICE)
                with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                    out = model(x)
                probs.append(torch.softmax(out.float(), 1).cpu())
            probs = torch.cat(probs)
            total = probs if total is None else total + probs
    return (total / len(TTA_OPS)).numpy()


def make_class_weights(y):
    if CLASS_WEIGHT_MODE == 'none':
        return None
    w = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y)
    if CLASS_WEIGHT_MODE == 'sqrt':
        w = np.sqrt(w)                       # 소수 클래스 우대 완화
        w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)


def train_fold(fold):
    tr_mask = fold_of_sample != fold
    va_mask = ~tr_mask
    X_tr, y_tr = data[tr_mask], labels[tr_mask]
    X_va, y_va = data[va_mask], labels[va_mask]

    # 소수 클래스 2x 오버샘플 (해당 fold의 train 부분만)
    sm = np.isin(y_tr, list(SMALL_CLASSES))
    X_tr_os = np.concatenate([X_tr, X_tr[sm]])
    y_tr_os = np.concatenate([y_tr, y_tr[sm]])
    print(f'\n--- fold {fold}: train {len(X_tr)}(+{sm.sum()} os) / val {len(X_va)} ---', flush=True)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(TSDataset(X_tr_os, y_tr_os, train=True),
                             batch_size=BATCH_SIZE, shuffle=True, generator=g,
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(TSDataset(X_va, y_va), batch_size=256, shuffle=False)

    model = build_model()
    cw = make_class_weights(y_tr_os)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    backbone = [p for n, p in model.named_parameters() if not n.startswith('classifier')]
    head = [p for n, p in model.named_parameters() if n.startswith('classifier')]
    optimizer = optim.AdamW([{'params': backbone, 'lr': LR_BACKBONE},
                             {'params': head, 'lr': LR_HEAD}], weight_decay=1e-4)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')

    best_f1, best_state, best_ep = 0.0, None, -1
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, y in val_loader:
                with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                    out = model(x.to(DEVICE))
                vp.append(out.float().argmax(1).cpu().numpy())
                vt.append(y.numpy())
        vp, vt = np.concatenate(vp), np.concatenate(vt)
        f1 = f1_score(vt, vp, average='macro')
        if f1 > best_f1:
            best_f1, best_ep = f1, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  [fold {fold}] Epoch {epoch+1:3d}/{EPOCHS} | '
              f'Val Acc {accuracy_score(vt, vp):.4f} F1 {f1:.4f} (best {best_f1:.4f} @ep{best_ep})',
              flush=True)

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT_DIR, f'best_model_fold{fold}.pth'))
    # OOF 예측 (TTA 적용)
    oof = predict_probs(model, X_va)
    oof_f1 = f1_score(y_va, oof.argmax(1), average='macro')
    print(f'  [fold {fold}] best(no-TTA) {best_f1:.4f} @ep{best_ep} | OOF(TTA) {oof_f1:.4f}', flush=True)
    return model, va_mask, oof, best_f1, oof_f1


# ---------------- CV 실행 ----------------
print(f'\n[3] Running {N_FOLDS}-fold CV (ConvNeXt-Tiny@{IMG_SIZE})...')
oof_probs = np.zeros((len(data), NUM_CLASSES), dtype=np.float32)
models, fold_best, fold_oof = [], [], []
for f in range(N_FOLDS):
    m, va_mask, oof, bf1, of1 = train_fold(f)
    oof_probs[va_mask] = oof
    models.append(m)
    fold_best.append(bf1)
    fold_oof.append(of1)

# ---------------- OOF 종합 분석 (26,010장 전체) ----------------
print('\n' + '=' * 70)
print('[4] OOF ANALYSIS (all 26,010 images, each predicted by a model that never saw it)')
print('=' * 70)
oof_pred = oof_probs.argmax(1)
oof_acc = accuracy_score(labels, oof_pred)
oof_macro = f1_score(labels, oof_pred, average='macro')
print(f'\n  Per-fold best (no-TTA): {["%.4f" % v for v in fold_best]}')
print(f'  Per-fold OOF   (TTA):   {["%.4f" % v for v in fold_oof]}')
print(f'  Fold mean +- std (TTA):  {np.mean(fold_oof):.4f} +- {np.std(fold_oof):.4f}')
print(f'\n  *** OOF overall: Acc {oof_acc:.4f}  Macro F1 {oof_macro:.4f} ***')
print(f'      (reference: iter6 real test Macro F1 = 0.9892)')

per_class = f1_score(labels, oof_pred, average=None)
order = np.argsort(per_class)
print('\n  Worst 12 per-class F1 (OOF):')
print(f'  {"cls":>4} {"n":>5} {"F1":>7} {"recall":>7} {"prec":>7}')
cm = confusion_matrix(labels, oof_pred, labels=range(NUM_CLASSES))
for c in order[:12]:
    n = cm[c].sum()
    rec = cm[c, c] / n if n else 0
    pcol = cm[:, c].sum()
    prec = cm[c, c] / pcol if pcol else 0
    print(f'  {c:>4} {n:>5} {per_class[c]:>7.4f} {rec:>7.4f} {prec:>7.4f}')

cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
pairs = [(cm_off[i, j], i, j) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if cm_off[i, j] > 0]
pairs.sort(reverse=True)
print(f'\n  Total OOF errors: {cm_off.sum()} / {len(labels)} ({cm_off.sum()/len(labels)*100:.2f}%)')
print('  Top 15 confusion pairs (true -> pred, count):')
for cnt, i, j in pairs[:15]:
    print(f'    {i:>2} -> {j:>2}: {cnt}')

print('\n  WATCH pairs (iter8에서 관측된 혼동):')
for i, j in WATCH_PAIRS:
    print(f'    {i:>2} -> {j:>2}: {cm[i, j]}')

np.save(os.path.join(OUT_DIR, 'oof_probs.npy'), oof_probs)
np.save(os.path.join(OUT_DIR, 'oof_labels.npy'), labels)

# ---------------- Test 추론 (3-fold 모델 앙상블) ----------------
print('\n[5] Test inference (3-fold model ensemble x 4 TTA)...')
test_dir = os.path.join(DATA_ROOT, 'Test')
result_df = pd.read_csv(TEMPLATE_RESULT)
test_imgs = []
for fname in result_df['id'].values:
    img = cv2.imread(os.path.join(test_dir, str(fname)))
    if img is None:
        test_imgs.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
test_imgs = np.array(test_imgs, dtype=np.uint8)

test_probs = None
for f, m in enumerate(models):
    p = predict_probs(m, test_imgs)
    test_probs = p if test_probs is None else test_probs + p
    print(f'  fold {f} done')
predictions = test_probs.argmax(1)
result_df['class'] = predictions.astype(int)
result_df.to_csv(os.path.join(OUT_DIR, 'result.csv'), index=False)
np.save(os.path.join(OUT_DIR, 'test_probs.npy'), test_probs)
print(f'  Saved {len(predictions)} predictions to result.csv')

# 분포 프록시 (진단 전용)
tc = np.array([np.sum(labels == i) for i in range(NUM_CLASSES)])
exp = tc / 3.0
pd_dist = pd.Series(predictions).value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
diff = pd_dist - exp
print(f'\n  Dist proxy: sum|diff|={np.abs(diff).sum():.0f}, max|diff|={np.abs(diff).max():.0f}, '
      f'min errors >= {np.abs(diff).sum()/2:.0f}')
print(f'    reference: iter6 78/12 (test 0.9892) | iter8 88/22 (미제출)')
worst = np.argsort(-np.abs(diff))[:6]
print('  Largest distribution deviations:')
for c in worst:
    print(f'    class {c:>2}: expected {exp[c]:>6.1f}, predicted {pd_dist[c]:>4} ({diff[c]:+.0f})')

print('\n' + '=' * 70)
print('[SUMMARY]')
print(f'  OOF Macro F1        : {oof_macro:.4f}   <- 신뢰 지표 (26,010장 전체)')
print(f'  OOF Accuracy        : {oof_acc:.4f}')
print(f'  Fold mean (TTA)     : {np.mean(fold_oof):.4f} +- {np.std(fold_oof):.4f}')
print(f'  Dist proxy sum|diff|: {np.abs(diff).sum():.0f} (iter6=78, iter8=88)')
print(f'  Dist proxy max|diff|: {np.abs(diff).max():.0f} (iter6=12, iter8=22)')
print(f'  11->30 OOF errors   : {cm[11, 30]}')
print('=' * 70)
print('[DONE]')
