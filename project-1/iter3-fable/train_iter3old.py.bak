# Iter3 (fable): Ensemble + hard-class focused augmentation
# Based on iter2 (Val Macro F1 0.9999) and its test-distribution analysis:
#  - Confusion-prone classes (11,12,20,21,30,32,34,38) get 2x oversampling
#    with a stronger augmentation pipeline
#  - 3-model ensemble (different seeds) x 4-op TTA, soft voting
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
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = r'C:\hong\python-ws\project-1\dataset\data 2'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(OUT_DIR, 'result.csv')
TEMPLATE_RESULT = r'C:\hong\python-ws\project-1\result.csv'
IMG_SIZE = 48
NUM_CLASSES = 43
BATCH_SIZE = 128
EPOCHS = 45
LR = 1e-3
SPLIT_SEED = 42          # fixed split so ensemble val F1 is measurable
ENSEMBLE_SEEDS = [42, 123, 777]
HARD_CLASSES = {11, 12, 20, 21, 30, 32, 34, 38}  # from iter2 distribution diff
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

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

# Hard-class 2x oversampling (each duplicate gets an independent random augment)
hard_mask = np.isin(y_train, list(HARD_CLASSES))
X_train_os = np.concatenate([X_train, X_train[hard_mask]])
y_train_os = np.concatenate([y_train, y_train[hard_mask]])
print(f'  After hard-class oversampling: {len(X_train_os)} (+{hard_mask.sum()})')

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_os), y=y_train_os)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# ---------------- Transforms ----------------
base_tf = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
])
# Stronger pipeline for confusion-prone classes
hard_tf = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.RandomAffine(degrees=18, translate=(0.15, 0.15), scale=(0.75, 1.25), shear=8),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
    T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
])
eval_tf = T.Compose([T.ToDtype(torch.float32, scale=True)])


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
            tf = hard_tf if int(y) in HARD_CLASSES else base_tf
        else:
            tf = eval_tf
        return tf(x), y


val_loader = DataLoader(TSDataset(X_val, y_val), batch_size=256, shuffle=False)


# ---------------- Model (same backbone as iter2) ----------------
def conv_block(cin, cout, drop):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout2d(drop),
    )


class Net(nn.Module):
    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, 0.2), conv_block(32, 64, 0.25),
            conv_block(64, 128, 0.3), conv_block(128, 256, 0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, n),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TSDataset(X_train_os, y_train_os, train=True),
                              batch_size=BATCH_SIZE, shuffle=True, generator=g)
    model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_f1, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, y in val_loader:
                vp.append(model(x.to(DEVICE)).argmax(1).cpu().numpy())
                vt.append(y.numpy())
        vp, vt = np.concatenate(vp), np.concatenate(vt)
        f1 = f1_score(vt, vp, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f'  [seed {seed}] Epoch {epoch+1:3d}/{EPOCHS} | '
                  f'Val Acc {accuracy_score(vt, vp):.4f} F1 {f1:.4f}', flush=True)

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


# ---------------- Train ensemble ----------------
print('\n[2] Training ensemble...')
models = [train_one(s) for s in ENSEMBLE_SEEDS]

# Ensemble validation score
val_tensor = torch.from_numpy(X_val).permute(0, 3, 1, 2).float() / 255.0
ens_val = sum(predict_probs(m, val_tensor) for m in models)
vp = ens_val.argmax(1).numpy()
print(f'\n[3] Ensemble Val Acc: {accuracy_score(y_val, vp):.4f} '
      f'Macro F1: {f1_score(y_val, vp, average="macro"):.4f}')

# ---------------- Test inference ----------------
print('\n[4] Test inference (3 models x 4 TTA)...')
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
test_tensor = torch.from_numpy(np.array(test_imgs, dtype=np.uint8)).permute(0, 3, 1, 2).float() / 255.0

probs = sum(predict_probs(m, test_tensor) for m in models)
predictions = probs.argmax(1).numpy()
result_df['class'] = predictions.astype(int)
result_df.to_csv(RESULT_PATH, index=False)
print(f'  Saved {len(predictions)} predictions to {RESULT_PATH}')

# Distribution check vs expected (train/3)
tc = np.array([np.sum(labels == i) for i in range(NUM_CLASSES)])
exp = tc / 3.0
pred_dist = pd.Series(predictions).value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
diff = pred_dist - exp
print(f'  Dist check: sum|diff|={np.abs(diff).sum():.0f} '
      f'(iter2: 184), max|diff|={np.abs(diff).max():.0f}, '
      f'min errors >= {np.abs(diff).sum()/2:.0f}')
print('[DONE]')
