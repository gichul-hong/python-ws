# Iter2 (fable): Improved CNN for GTSRB traffic sign classification
# Improvements over iter1:
#  - 48x48 input resolution (vs 30x30)
#  - Data augmentation (rotation, translation, scale, color jitter)
#  - Best model selected by val macro F1 (vs accuracy)
#  - AdamW + label smoothing + cosine annealing
#  - Test-time augmentation (TTA) at inference
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
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = r'C:\hong\python-ws\project-1\dataset\data 2'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(OUT_DIR, 'result.csv')
TEMPLATE_RESULT = r'C:\hong\python-ws\project-1\result.csv'
MODEL_PATH = os.path.join(OUT_DIR, 'best_model.pth')
IMG_SIZE = 48
NUM_CLASSES = 43
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
print(f'Device: {DEVICE}')

# ---------------- Data loading ----------------
print('\n[1] Loading training data...')
data, labels = [], []
train_path = os.path.join(DATA_ROOT, 'Train')
for i in range(NUM_CLASSES):
    class_dir = os.path.join(train_path, str(i))
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        if not fname.endswith('.png'):
            continue
        img = cv2.imread(os.path.join(class_dir, fname))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        data.append(img)
        labels.append(i)

data = np.array(data, dtype=np.uint8)
labels = np.array(labels, dtype=np.int64)
print(f'  Total images: {len(data)}')

X_train, X_val, y_train, y_val = train_test_split(
    data, labels, test_size=0.2, random_state=SEED, stratify=labels)
print(f'  Train: {len(X_train)}, Val: {len(X_val)}')

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# ---------------- Dataset ----------------
train_tf = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
])
eval_tf = T.Compose([T.ToDtype(torch.float32, scale=True)])


class TSDataset(Dataset):
    def __init__(self, images, labels, tf):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.from_numpy(labels)
        self.tf = tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.tf(self.images[idx]), self.labels[idx]


train_loader = DataLoader(TSDataset(X_train, y_train, train_tf),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(TSDataset(X_val, y_val, eval_tf),
                        batch_size=256, shuffle=False, num_workers=0)


# ---------------- Model ----------------
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
            conv_block(3, 32, 0.2),    # 48 -> 24
            conv_block(32, 64, 0.25),  # 24 -> 12
            conv_block(64, 128, 0.3),  # 12 -> 6
            conv_block(128, 256, 0.3), # 6 -> 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, n),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = Net().to(DEVICE)
print(f'\n[2] Model params: {sum(p.numel() for p in model.parameters()):,}')

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ---------------- Training ----------------
print('\n[3] Training...')
best_f1, best_state = 0.0, None
for epoch in range(EPOCHS):
    model.train()
    tl, tp, tt = 0.0, [], []
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tl += loss.item() * x.size(0)
        tp.append(out.argmax(1).cpu().numpy())
        tt.append(y.cpu().numpy())
    scheduler.step()
    tp, tt = np.concatenate(tp), np.concatenate(tt)

    model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x.to(DEVICE))
            vp.append(out.argmax(1).cpu().numpy())
            vt.append(y.numpy())
    vp, vt = np.concatenate(vp), np.concatenate(vt)
    val_acc = accuracy_score(vt, vp)
    val_f1 = f1_score(vt, vp, average='macro')

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f'  Epoch {epoch+1:3d}/{EPOCHS} | Train Loss {tl/len(tt):.4f} '
              f'Acc {accuracy_score(tt, tp):.4f} | Val Acc {val_acc:.4f} F1 {val_f1:.4f}')

model.load_state_dict(best_state)
torch.save(best_state, MODEL_PATH)
print(f'\n  Best Val Macro F1: {best_f1:.4f}')

# ---------------- Test inference with TTA ----------------
print('\n[4] Test inference (TTA)...')
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

# TTA: identity, small +/- rotations, slight scale via affine
import torchvision.transforms.functional as TF
tta_ops = [
    lambda b: b,
    lambda b: TF.rotate(b, 8),
    lambda b: TF.rotate(b, -8),
    lambda b: TF.affine(b, angle=0, translate=[2, 2], scale=1.1, shear=[0.0]),
]

model.eval()
probs_sum = None
with torch.no_grad():
    for op in tta_ops:
        probs = []
        for i in range(0, len(test_tensor), 256):
            batch = op(test_tensor[i:i+256]).to(DEVICE)
            probs.append(torch.softmax(model(batch), 1).cpu())
        probs = torch.cat(probs)
        probs_sum = probs if probs_sum is None else probs_sum + probs

predictions = probs_sum.argmax(1).numpy()
result_df['class'] = predictions.astype(int)
result_df.to_csv(RESULT_PATH, index=False)
print(f'  Saved {len(predictions)} predictions to {RESULT_PATH}')
print('[DONE]')
