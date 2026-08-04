# Iter8 (deepseek): Track-aware split + Heterogeneous ensemble
# 배경: iter6 val F1 0.9999 vs 실제 test F1 0.9892 → random split이 같은 트랙(동일 표지판
#       30연속 프레임)을 train/val 양쪽에 넣어 val이 과대평가됨을 확인.
# 핵심 변경:
#  A. Track-aware split: 파일명 {class}_{track}_{frame}.png 에서 트랙 단위로 분할
#     → 같은 물리적 표지판은 train/val 한쪽에만 존재 (test 조건과 동일한 정직한 val)
#  B. 이종 앙상블: ResNet-50@96 + EfficientNet-B0@128 + ConvNeXt-Tiny@96
#     (seed 다양성 대신 아키텍처/해상도 다양성)
#  C. 혼동 클래스 분석: track-aware val 기반 confusion 리포트 자동 출력
#  D. 도메인 강건성 증강: 광도 변화 강화 + motion blur + 저해상도 시뮬레이션
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
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = r'C:\hong\python-ws\project-1\dataset\data 2'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(OUT_DIR, 'result.csv')
TEMPLATE_RESULT = r'C:\hong\python-ws\project-1\result.csv'
LOAD_SIZE = 128          # 원본은 최대 해상도로 1회 로드, 모델별로 리사이즈
NUM_CLASSES = 43
EPOCHS = 30
WARMUP_EPOCHS = 3
LR_BACKBONE = 3e-4
LR_HEAD = 1e-3
SPLIT_SEED = 42
VAL_TRACK_RATIO = 0.2    # 클래스별 트랙의 20%를 val로
SMALL_CLASSES = {0, 6, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 34, 36, 37, 39, 40, 41, 42}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
print(f'Device: {DEVICE}')

# 이종 앙상블 구성: (이름, 빌더, 입력 해상도, 배치)
MODEL_CONFIGS = [
    ('resnet50_96',      'resnet50',      96,  96),
    ('efficientnet_128', 'efficientnet',  128, 64),
    ('convnext_96',      'convnext',      96,  64),
]

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
                print(f'  [WARN] unexpected filename: {fname} (class {i}) — track parsing failed')
            continue
        img = cv2.imread(os.path.join(class_dir, fname))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(cv2.resize(img, (LOAD_SIZE, LOAD_SIZE), interpolation=cv2.INTER_AREA))
        labels.append(i)
        tracks.append((i, int(m.group(2))))   # (class, track) = 고유 트랙 키

data = np.array(data, dtype=np.uint8)
labels = np.array(labels, dtype=np.int64)
print(f'  Total: {len(data)} images, {len(set(tracks))} unique tracks')

# ---------------- A. Track-aware split ----------------
# 같은 트랙(동일 물리 표지판)은 train/val 중 한쪽에만 → val이 test 조건을 근사
print('\n[2] Track-aware split (no track overlap between train/val)...')
rng = np.random.RandomState(SPLIT_SEED)
class_tracks = defaultdict(list)
for t in set(tracks):
    class_tracks[t[0]].append(t)

val_track_set = set()
for c in range(NUM_CLASSES):
    ts = sorted(class_tracks[c], key=lambda x: x[1])
    n_val = max(1, int(round(len(ts) * VAL_TRACK_RATIO)))
    picked = rng.choice(len(ts), size=n_val, replace=False)
    val_track_set.update(ts[j] for j in picked)

is_val = np.array([t in val_track_set for t in tracks])
X_train, y_train = data[~is_val], labels[~is_val]
X_val, y_val = data[is_val], labels[is_val]
print(f'  Train: {len(X_train)} ({len(set(tracks)) - len(val_track_set)} tracks), '
      f'Val: {len(X_val)} ({len(val_track_set)} tracks)')

# Small-class 2x oversampling (train만)
small_mask = np.isin(y_train, list(SMALL_CLASSES))
X_train_os = np.concatenate([X_train, X_train[small_mask]])
y_train_os = np.concatenate([y_train, y_train[small_mask]])
print(f'  After small-class oversampling: {len(X_train_os)} (+{small_mask.sum()})')

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_os), y=y_train_os)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# ---------------- D. 도메인 강건성 증강 (flip 금지) ----------------
def make_train_tf(img_size, strong=False):
    """광도 강화 + motion blur + 저해상도 시뮬레이션. strong=소수클래스용."""
    deg, tr, sc, sh = (18, 0.15, (0.75, 1.25), 8) if strong else (12, 0.1, (0.85, 1.15), 5)
    ops = [
        T.ToDtype(torch.float32, scale=True),
        T.RandomAffine(degrees=deg, translate=(tr, tr), scale=sc, shear=sh),
        # 광도: test 촬영조건 변화 대응 (어두움/과노출/저채도)
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3,
                      hue=0.05 if strong else 0.02),
        # motion blur / defocus 시뮬레이션
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.35),
        # 저해상도 시뮬레이션 (작게 줄였다 복원 → 원거리 촬영 근사)
        T.RandomApply([T.Compose([
            T.Resize(img_size // 2, antialias=True),
            T.Resize(img_size, antialias=True),
        ])], p=0.25),
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if strong:
        ops.append(T.RandomErasing(p=0.2, scale=(0.02, 0.08)))  # 픽토그램 보존 위해 축소
    return T.Compose(ops)


def make_eval_tf(img_size):
    return T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TSDataset(Dataset):
    def __init__(self, images, labels, img_size, train=False):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.from_numpy(labels)
        self.train = train
        self.base_tf = make_train_tf(img_size, strong=False)
        self.small_tf = make_train_tf(img_size, strong=True)
        self.eval_tf = make_eval_tf(img_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x, y = self.images[idx], self.labels[idx]
        if self.train:
            tf = self.small_tf if int(y) in SMALL_CLASSES else self.base_tf
        else:
            tf = self.eval_tf
        return tf(x), y


# ---------------- B. 이종 모델 빌더 ----------------
def build_model(arch):
    if arch == 'resnet50':
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, NUM_CLASSES))
        head_prefix = 'fc'
    elif arch == 'efficientnet':
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, NUM_CLASSES))
        head_prefix = 'classifier'
    elif arch == 'convnext':
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(768, NUM_CLASSES)
        head_prefix = 'classifier'
    else:
        raise ValueError(arch)
    return m.to(DEVICE), head_prefix


def train_one(name, arch, img_size, batch_size, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TSDataset(X_train_os, y_train_os, img_size, train=True),
                              batch_size=batch_size, shuffle=True, generator=g,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(TSDataset(X_val, y_val, img_size),
                            batch_size=256, shuffle=False)
    model, head_prefix = build_model(arch)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith(head_prefix)]
    head_params = [p for n, p in model.named_parameters() if n.startswith(head_prefix)]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params, 'lr': LR_HEAD},
    ], weight_decay=1e-4)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')

    best_f1, best_state = 0.0, None
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
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  [{name}] Epoch {epoch+1:3d}/{EPOCHS} | '
              f'Val Acc {accuracy_score(vt, vp):.4f} F1 {f1:.4f} (best {best_f1:.4f})', flush=True)

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT_DIR, f'best_model_{name}.pth'))
    print(f'  [{name}] Best track-aware Val Macro F1: {best_f1:.4f}', flush=True)
    return model


TTA_OPS = [
    lambda b: b,
    lambda b: TF.rotate(b, 8),
    lambda b: TF.rotate(b, -8),
    lambda b: TF.affine(b, angle=0, translate=[2, 2], scale=1.1, shear=[0.0]),
]


def predict_probs(model, images_uint8, img_size, batch=256):
    """images_uint8: (N,H,W,3). 모델별 해상도로 리사이즈+정규화 후 TTA 평균."""
    eval_tf = make_eval_tf(img_size)
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
    return total / len(TTA_OPS)


# ---------------- Train heterogeneous ensemble ----------------
print('\n[3] Training heterogeneous ensemble...')
models = []
for name, arch, img_size, bs in MODEL_CONFIGS:
    print(f'\n--- {name} (arch={arch}, {img_size}px, batch={bs}) ---')
    models.append((name, train_one(name, arch, img_size, bs), img_size))

# ---------------- Ensemble validation + C. confusion 분석 ----------------
print('\n[4] Ensemble validation (track-aware val)...')
ens_val = None
for name, m, img_size in models:
    p = predict_probs(m, X_val, img_size)
    solo = f1_score(y_val, p.argmax(1).numpy(), average='macro')
    print(f'  [{name}] solo Val Macro F1 (TTA): {solo:.4f}')
    ens_val = p if ens_val is None else ens_val + p
vp = ens_val.argmax(1).numpy()
ens_acc = accuracy_score(y_val, vp)
ens_f1 = f1_score(y_val, vp, average='macro')
print(f'\n  Ensemble Val Acc: {ens_acc:.4f} Macro F1: {ens_f1:.4f}')

# 클래스별 F1 하위 8개 + 상위 혼동 쌍 리포트
per_class = f1_score(y_val, vp, average=None)
worst = np.argsort(per_class)[:8]
print('\n  Worst per-class F1 (track-aware val):')
for c in worst:
    print(f'    class {c:2d}: F1 {per_class[c]:.4f}')
cm = confusion_matrix(y_val, vp, labels=range(NUM_CLASSES))
np.fill_diagonal(cm, 0)
pairs = [(cm[i, j], i, j) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if cm[i, j] > 0]
pairs.sort(reverse=True)
print('  Top confusion pairs (true -> pred, count):')
for cnt, i, j in pairs[:10]:
    print(f'    {i:2d} -> {j:2d}: {cnt}')

# ---------------- Test inference ----------------
print('\n[5] Test inference (3-arch ensemble x 4 TTA)...')
test_dir = os.path.join(DATA_ROOT, 'Test')
result_df = pd.read_csv(TEMPLATE_RESULT)
test_imgs = []
for fname in result_df['id'].values:
    img = cv2.imread(os.path.join(test_dir, str(fname)))
    if img is None:
        test_imgs.append(np.zeros((LOAD_SIZE, LOAD_SIZE, 3), dtype=np.uint8))
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_imgs.append(cv2.resize(img, (LOAD_SIZE, LOAD_SIZE), interpolation=cv2.INTER_AREA))
test_imgs = np.array(test_imgs, dtype=np.uint8)

probs = None
for name, m, img_size in models:
    p = predict_probs(m, test_imgs, img_size)
    probs = p if probs is None else probs + p
predictions = probs.argmax(1).numpy()
result_df['class'] = predictions.astype(int)
result_df.to_csv(RESULT_PATH, index=False)
print(f'  Saved {len(predictions)} predictions to {RESULT_PATH}')

# 앙상블 확률 저장 (사후 분석/타 iteration과의 soft voting 결합용)
np.save(os.path.join(OUT_DIR, 'test_probs.npy'), probs.numpy())

# Distribution check (진단 전용)
tc = np.array([np.sum(labels == i) for i in range(NUM_CLASSES)])
exp = tc / 3.0
pred_dist = pd.Series(predictions).value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
diff = pred_dist - exp
print(f'  Dist check: sum|diff|={np.abs(diff).sum():.0f} '
      f'(iter6: 78 -> test F1 0.9892), max|diff|={np.abs(diff).max():.0f}, '
      f'min errors >= {np.abs(diff).sum()/2:.0f}')
print('[DONE]')
