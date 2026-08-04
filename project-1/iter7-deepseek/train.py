# Iter7 (deepseek): Full-data ResNet-50 3-seed + iter4 custom CNN heterogeneous ensemble
# - ALL 26,010 training images (no val split, hyperparameters confirmed by iter6)
# - ResNet-50 (IMAGENET1K_V2), 96x96, discriminative LR, AMP, 30 epochs
# - 3 seeds (42, 123, 777) x 4-op TTA soft voting
# - Heterogeneous ensemble: 3x ResNet-50 + 3x iter4 custom CNN (5-model total)
# - Distribution check is diagnostic ONLY
import os, sys
import numpy as np
import cv2
import pandas as pd
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
TEMPLATE_RESULT = r'C:\hong\python-ws\project-1\result.csv'
ITER4_DIR = r'C:\hong\python-ws\project-1\iter4-zai'
IMG_SIZE = 96
NUM_CLASSES = 43
BATCH_SIZE = 96
EPOCHS = 30
WARMUP_EPOCHS = 3
LR_BACKBONE = 3e-4
LR_HEAD = 1e-3
ENSEMBLE_SEEDS = [42, 123, 777]
SMALL_CLASSES = {0, 6, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 34, 36, 37, 39, 40, 41, 42}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
print(f'Device: {DEVICE}')

# ---------------- Data loading (ALL 26,010 images, no split) ----------------
print('\n[1] Loading ALL training data (no val split)...')
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

X_all = np.array(data, dtype=np.uint8)
y_all = np.array(labels, dtype=np.int64)
print(f'  Train: {len(X_all)} (all data, no val split)')

# Small-class 2x oversampling
small_mask = np.isin(y_all, list(SMALL_CLASSES))
X_train_os = np.concatenate([X_all, X_all[small_mask]])
y_train_os = np.concatenate([y_all, y_all[small_mask]])
print(f'  After small-class oversampling: {len(X_train_os)} (+{small_mask.sum()})')

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_os), y=y_train_os)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# ---------------- Transforms (NO horizontal flip) ----------------
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
eval_tf_rn = T.Compose([T.ToDtype(torch.float32, scale=True), normalize])


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
            tf = eval_tf_rn
        return tf(x), y


# ---------------- ResNet-50 model ----------------
def build_rn50():
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, NUM_CLASSES))
    return m.to(DEVICE)


def train_one_rn(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TSDataset(X_train_os, y_train_os, train=True),
                              batch_size=BATCH_SIZE, shuffle=True, generator=g,
                              num_workers=0, pin_memory=True)
    model = build_rn50()
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

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f'  [seed {seed}] Epoch {epoch+1:3d}/{EPOCHS} | '
                  f'Loss {running_loss/len(train_loader):.4f}', flush=True)

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f'best_model_seed{seed}.pth'))
    print(f'  [seed {seed}] Saved final weights (full-data {EPOCHS} epochs)', flush=True)
    return model


TTA_OPS = [
    lambda b: b,
    lambda b: TF.rotate(b, 8),
    lambda b: TF.rotate(b, -8),
    lambda b: TF.affine(b, angle=0, translate=[2, 2], scale=1.1, shear=[0.0]),
]


def predict_probs_rn(model, tensor):
    """tensor: ImageNet-normalized float tensor (N,3,H,W) on CPU."""
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


def to_norm_tensor_rn(images_uint8):
    t = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float() / 255.0
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


# ---------------- Iter4 custom CNN (for heterogeneous ensemble) ----------------
# Exact architecture from iter4-zai/train.py, 64x64 input
def conv_block_4(cin, cout, drop):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout2d(drop),
    )


class Net4(nn.Module):
    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            conv_block_4(3, 32, 0.2), conv_block_4(32, 64, 0.25),
            conv_block_4(64, 128, 0.3), conv_block_4(128, 256, 0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, n),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_iter4_models():
    models4 = []
    for seed in ENSEMBLE_SEEDS:
        m = Net4().to(DEVICE)
        m.load_state_dict(torch.load(os.path.join(ITER4_DIR, f'best_model_seed{seed}.pth'),
                                     map_location=DEVICE, weights_only=True))
        m.eval()
        models4.append(m)
        print(f'  Loaded iter4 model seed{seed}')
    return models4


def predict_probs_4(model, tensor):
    """tensor: raw scaled float tensor (N,3,H,W) on CPU (no ImageNet norm)."""
    model.eval()
    total = None
    with torch.no_grad():
        for op in TTA_OPS:
            probs = []
            for i in range(0, len(tensor), 256):
                batch = op(tensor[i:i+256]).to(DEVICE)
                probs.append(torch.softmax(model(batch), 1).cpu())
            probs = torch.cat(probs)
            total = probs if total is None else total + probs
    return total / len(TTA_OPS)


# ---------------- Train 3-seed ResNet-50 ensemble ----------------
print('\n[2] Training 3-seed ResNet-50 (full 26,010 data)...')
models_rn = [train_one_rn(s) for s in ENSEMBLE_SEEDS]

# ---------------- Test inference: ResNet-50 ----------------
print('\n[3] Test inference: 3-seed ResNet-50 ensemble (96x96, ImageNet norm)...')
test_dir = os.path.join(DATA_ROOT, 'Test')
result_df = pd.read_csv(TEMPLATE_RESULT)
test_imgs_96 = []
for fname in result_df['id'].values:
    img = cv2.imread(os.path.join(test_dir, str(fname)))
    if img is None:
        test_imgs_96.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_imgs_96.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
test_rn_tensor = to_norm_tensor_rn(np.array(test_imgs_96, dtype=np.uint8))

probs_rn = sum(predict_probs_rn(m, test_rn_tensor) for m in models_rn)
preds_rn = probs_rn.argmax(1).numpy()
result_df['class'] = preds_rn.astype(int)
RESULT_RN_PATH = os.path.join(OUT_DIR, 'result_resnet.csv')
result_df.to_csv(RESULT_RN_PATH, index=False)
print(f'  Saved {len(preds_rn)} predictions (3-seed ResNet-50) to {RESULT_RN_PATH}')

# Distribution check
tc = np.array([np.sum(labels == i) for i in range(NUM_CLASSES)])
exp = tc / 3.0
pred_dist_rn = pd.Series(preds_rn).value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
diff_rn = pred_dist_rn - exp
print(f'  Dist check (ResNet-50): sum|diff|={np.abs(diff_rn).sum():.0f} '
      f'(iter6: 78), max|diff|={np.abs(diff_rn).max():.0f}, '
      f'min errors >= {np.abs(diff_rn).sum()/2:.0f}')

# ---------------- Heterogeneous ensemble: ResNet-50 + iter4 custom CNN ----------------
print('\n[4] Heterogeneous ensemble (3x ResNet-50 + 3x iter4 custom CNN)...')
models_4 = load_iter4_models()

# Test images at 64x64 for iter4 CNN (no ImageNet norm)
test_imgs_64 = []
for fname in result_df['id'].values:
    img = cv2.imread(os.path.join(test_dir, str(fname)))
    if img is None:
        test_imgs_64.append(np.zeros((64, 64, 3), dtype=np.uint8))
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_imgs_64.append(cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA))
test_4_tensor = torch.from_numpy(np.array(test_imgs_64, dtype=np.uint8)).permute(0, 3, 1, 2).float() / 255.0

probs_4 = sum(predict_probs_4(m, test_4_tensor) for m in models_4)

# Combine: equal weight for all 6 models
probs_hetero = probs_rn + probs_4
preds_hetero = probs_hetero.argmax(1).numpy()
result_df['class'] = preds_hetero.astype(int)
RESULT_HETERO_PATH = os.path.join(OUT_DIR, 'result_hetero.csv')
result_df.to_csv(RESULT_HETERO_PATH, index=False)
print(f'  Saved {len(preds_hetero)} predictions (heterogeneous ensemble) to {RESULT_HETERO_PATH}')

pred_dist_hetero = pd.Series(preds_hetero).value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
diff_hetero = pred_dist_hetero - exp
print(f'  Dist check (Heterogeneous): sum|diff|={np.abs(diff_hetero).sum():.0f} '
      f'max|diff|={np.abs(diff_hetero).max():.0f}, '
      f'min errors >= {np.abs(diff_hetero).sum()/2:.0f}')

# Also save the "default" result.csv as the heterogeneous ensemble (best overall)
result_df.to_csv(os.path.join(OUT_DIR, 'result.csv'), index=False)
print(f'  Default result.csv = heterogeneous ensemble')

print('\n[5] Summary:')
print(f'  ResNet-50 (3-seed):     sum|diff|={np.abs(diff_rn).sum():.0f}, max|diff|={np.abs(diff_rn).max():.0f}')
print(f'  Heterogeneous (6-model): sum|diff|={np.abs(diff_hetero).sum():.0f}, max|diff|={np.abs(diff_hetero).max():.0f}')
print('[DONE]')