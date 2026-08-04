# Iter6 (fable): ResNet-50 (ImageNet pretrained) full fine-tune, per paper 2503.06313
# - ResNet-50 IMAGENET1K_V2, ALL layers trainable, 96x96, ImageNet normalization
# - Discriminative LR: backbone 3e-4, head 1e-3, AdamW + 3-epoch warmup + cosine
# - AMP mixed precision for speed (RTX 2060)
# - Small-class (train-only criterion, 22 classes @180 imgs) 2x oversampling + strong aug
# - NO horizontal flip (traffic signs are asymmetric)
# - 2-seed ensemble x 4-op TTA soft voting
# - Distribution check is diagnostic ONLY (no test reference in training)
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torchvision.models import resnet50, ResNet50_Weights
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = r'C:\hong\python-ws\project-1\dataset\data 2'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(OUT_DIR, 'result.csv')
TEMPLATE_RESULT = r'C:\hong\python-ws\project-1\result.csv'
IMG_SIZE = 96
NUM_CLASSES = 43
BATCH_SIZE = 96
EPOCHS = 30
WARMUP_EPOCHS = 3
LR_BACKBONE = 3e-4
LR_HEAD = 1e-3
SPLIT_SEED = 42
ENSEMBLE_SEEDS = [42, 123]
# Small classes: 180 training images each (train-only criterion, iter4 justification)
SMALL_CLASSES = {0, 6, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 34, 36, 37, 39, 40, 41, 42}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------- Data loading ----------------
print('\n[1] Loading training data...')
data, labels = [], []
train_path = os.path.join(DATA_ROOT, 'Train')
for i in range(NUM_CLASSES):
    class_dir = os.path.join(train_path, str(i))
    for fname in os.listdir(class_dir):
        if not fname.endswith('.png'):
            continue
        img = cv2.imread(os.path.join(class_dir, fname))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
        labels.append(i)

data = np.array(data, dtype=np.uint8)
labels = np.array(labels, dtype=np.int64)
X_train, X_val, y_train, y_val = train_test_split(
    data, labels, test_size=0.2, random_state=SPLIT_SEED, stratify=labels)
print(f'  Train: {len(X_train)}, Val: {len(X_val)}')

# Small-class 2x oversampling (each duplicate gets independent random augment)
small_mask = np.isin(y_train, list(SMALL_CLASSES))
X_train_os = np.concatenate([X_train, X_train[small_mask]])
y_train_os = np.concatenate([y_train, y_train[small_mask]])
print(f'  After small-class oversampling: {len(X_train_os)} (+{small_mask.sum()})')

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_os), y=y_train_os)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# ---------------- Transforms (NO horizontal flip: signs are asymmetric) ----------------
normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
base_tf = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    normalize,
])
small_tf = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.RandomAffine(degrees=18, translate=(0.15, 0.15), scale=(0.75, 1.25), shear=8),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
    normalize,
    T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
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


val_loader = DataLoader(TSDataset(X_val, y_val), batch_size=256, shuffle=False)


# ---------------- Model ----------------
def build_model():
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, NUM_CLASSES))
    return m.to(DEVICE)


def train_one(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TSDataset(X_train_os, y_train_os, train=True),
                              batch_size=BATCH_SIZE, shuffle=True, generator=g,
                              num_workers=0, pin_memory=True)
    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith('fc')]
    head_params = [p for n, p in model.named_parameters() if n.startswith('fc')]
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
        print(f'  [seed {seed}] Epoch {epoch+1:3d}/{EPOCHS} | '
              f'Val Acc {accuracy_score(vt, vp):.4f} F1 {f1:.4f} (best {best_f1:.4f})', flush=True)

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT_DIR, f'best_model_seed{seed}.pth'))
    print(f'  [seed {seed}] Best Val Macro F1: {best_f1:.4f}', flush=True)
    return model


TTA_OPS = [
    lambda b: b,
    lambda b: TF.rotate(b, 8),
    lambda b: TF.rotate(b, -8),
    lambda b: TF.affine(b, angle=0, translate=[2, 2], scale=1.1, shear=[0.0]),
]


def predict_probs(model, tensor):
    """tensor: normalized float tensor (N,3,H,W) on CPU."""
    model.eval()
    total = None
    with torch.no_grad():
        for op in TTA_OPS:
            probs = []
            for i in range(0, len(tensor), 256):
                batch = op(tensor[i:i+256]).to(DEVICE)
                with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                    out = model(batch)
                probs.append(torch.softmax(out.float(), 1).cpu())
            probs = torch.cat(probs)
            total = probs if total is None else total + probs
    return total / len(TTA_OPS)


def to_normalized_tensor(images_uint8):
    t = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float() / 255.0
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


# ---------------- Train ensemble ----------------
print('\n[2] Training ensemble (ResNet-50 fine-tune)...')
models = [train_one(s) for s in ENSEMBLE_SEEDS]

# Ensemble validation score
val_tensor = to_normalized_tensor(X_val)
ens_val = sum(predict_probs(m, val_tensor) for m in models)
vp = ens_val.argmax(1).numpy()
print(f'\n[3] Ensemble Val Acc: {accuracy_score(y_val, vp):.4f} '
      f'Macro F1: {f1_score(y_val, vp, average="macro"):.4f}')

# ---------------- Test inference ----------------
print('\n[4] Test inference (2 models x 4 TTA)...')
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
test_tensor = to_normalized_tensor(np.array(test_imgs, dtype=np.uint8))

probs = sum(predict_probs(m, test_tensor) for m in models)
predictions = probs.argmax(1).numpy()
result_df['class'] = predictions.astype(int)
result_df.to_csv(RESULT_PATH, index=False)
print(f'  Saved {len(predictions)} predictions to {RESULT_PATH}')

# Distribution check vs expected (train/3) -- diagnostic ONLY
tc = np.array([np.sum(labels == i) for i in range(NUM_CLASSES)])
exp = tc / 3.0
pred_dist = pd.Series(predictions).value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
diff = pred_dist - exp
print(f'  Dist check: sum|diff|={np.abs(diff).sum():.0f} '
      f'(iter2: 184, iter3: 74, iter4: 96), max|diff|={np.abs(diff).max():.0f}, '
      f'min errors >= {np.abs(diff).sum()/2:.0f}')
print('[DONE]')
