# Iter5 (zai): ResNet18 frozen feature extractor + ML classifiers
# - ResNet18 ImageNet pretrained, FROZEN (no finetune), 224x224, avgpool 512-dim features
# - Feature extraction: 1x base + augmented variants for oversampled small classes
# - ML algorithms: XGBoost (primary), Random Forest, Linear SVM, RBF-SVM
# - Small-class 2x oversampling (train-only criterion, NO test reference)
# - Ensemble: ML soft voting + comparison
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torchvision.models import resnet18, ResNet18_Weights
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('[WARN] xgboost not available, skipping XGBoost')

DATA_ROOT = r'C:\hong\python-ws\project-1\dataset\data 2'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(OUT_DIR, 'result.csv')
TEMPLATE_RESULT = r'C:\hong\python-ws\project-1\result.csv'
FEAT_SIZE = 224        # ResNet expects ~224 for ImageNet distribution
NUM_CLASSES = 43
SPLIT_SEED = 42
# Small classes: 180 training images each (train-only criterion, no test reference)
SMALL_CLASSES = {0, 6, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 34, 36, 37, 39, 40, 41, 42}
# Augmented feature variants per oversampled small-class sample
N_AUG_SMALL = 1
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
        data.append(cv2.resize(img, (FEAT_SIZE, FEAT_SIZE), interpolation=cv2.INTER_AREA))
        labels.append(i)

data = np.array(data, dtype=np.uint8)
labels = np.array(labels, dtype=np.int64)
X_train, X_val, y_train, y_val = train_test_split(
    data, labels, test_size=0.2, random_state=SPLIT_SEED, stratify=labels)
print(f'  Train: {len(X_train)}, Val: {len(X_val)}')

# ---------------- Transforms ----------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
eval_tf = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
# Light augmentation for oversampled small-class feature extraction (adds diversity)
aug_tf = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=4),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
])


def extract_features(model, images_uint8, transform, batch_size=64):
    """Extract avgpool features (512-dim) from a frozen ResNet."""
    model.eval()
    feats = []
    n = len(images_uint8)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = images_uint8[i:i+batch_size]
            t = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous()
            t = transform(t)
            t = t.to(DEVICE)
            f = model(t)               # (B, 512)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------- Build frozen ResNet18 ----------------
print('\n[2] Building frozen ResNet18 feature extractor...')
weights = ResNet18_Weights.IMAGENET1K_V1
rn = resnet18(weights=weights)
rn.fc = nn.Identity()    # output = avgpool 512-dim
rn = rn.to(DEVICE).eval()
for p in rn.parameters():
    p.requires_grad = False

# ---------------- Feature extraction (with caching) ----------------
print('\n[3] Extracting features (frozen ResNet18, 224x224)...')
CACHE_DIR = os.path.join(OUT_DIR, '_feat_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
npz = os.path.join(CACHE_DIR, 'features.npz')
if os.path.exists(npz):
    d = np.load(npz)
    F_train, F_val = d['F_train'], d['F_val']
    X_small = X_train[np.isin(y_train, list(SMALL_CLASSES))]
    y_small = y_train[np.isin(y_train, list(SMALL_CLASSES))]
    print('  Loaded cached features')
else:
    F_train = extract_features(rn, X_train, eval_tf)
    F_val = extract_features(rn, X_val, eval_tf)
    np.savez(npz, F_train=F_train, F_val=F_val)
    X_small = X_train[np.isin(y_train, list(SMALL_CLASSES))]
    y_small = y_train[np.isin(y_train, list(SMALL_CLASSES))]
    print('  Saved feature cache')
print(f'  Train features: {F_train.shape}, Val features: {F_val.shape}')

F_aug_list = [F_train]
y_aug_list = [y_train]
for k in range(N_AUG_SMALL):
    aug_npz = os.path.join(CACHE_DIR, f'feat_aug_{k}.npz')
    if os.path.exists(aug_npz):
        F_aug_k = np.load(aug_npz)['F']
    else:
        F_aug_k = extract_features(rn, X_small, aug_tf)
        np.savez(aug_npz, F=F_aug_k)
    F_aug_list.append(F_aug_k)
    y_aug_list.append(y_small)
print(f'  Small-class samples to oversample: {len(y_small)}')
F_train_os = np.concatenate(F_aug_list)
y_train_os = np.concatenate(y_aug_list)
print(f'  After oversampling: {F_train_os.shape}')

# Standardize features for SVM/RF
scaler = StandardScaler()
F_train_os_s = scaler.fit_transform(F_train_os)
F_val_s = scaler.transform(F_val)

# ---------------- ML classifiers ----------------
print('\n[4] Training ML classifiers...')

results = {}
preds_val = {}

# Class weight for imbalance
cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_train_os)
cw_dict = {i: w for i, w in enumerate(cw)}
# XGBoost sample weight array
sw = np.array([cw_dict[c] for c in y_train_os])

# --- XGBoost ---
if HAS_XGB:
    print('  [XGBoost] training...')
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.2,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        objective='multi:softmax', num_class=NUM_CLASSES,
        tree_method='hist', n_jobs=-1, random_state=42,
        eval_metric='mlogloss', verbosity=0,
        early_stopping_rounds=20,
    )
    xgb.fit(F_train_os, y_train_os, sample_weight=sw,
            eval_set=[(F_val, y_val)], verbose=False)
    pv = xgb.predict(F_val)
    acc = accuracy_score(y_val, pv)
    f1 = f1_score(y_val, pv, average='macro')
    print(f'  [XGBoost] Val Acc {acc:.4f} F1 {f1:.4f} (best_iter={xgb.best_iteration})')
    results['xgb'] = f1
    preds_val['xgb'] = pv
    # save predict_proba for ensemble
    try:
        xgb_proba = xgb.predict_proba(F_val)
    except Exception:
        xgb_proba = None

# --- Random Forest ---
print('  [RandomForest] training...')
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=1,
    class_weight='balanced', n_jobs=-1, random_state=42,
)
rf.fit(F_train_os_s, y_train_os)
pv = rf.predict(F_val_s)
acc = accuracy_score(y_val, pv)
f1 = f1_score(y_val, pv, average='macro')
print(f'  [RandomForest] Val Acc {acc:.4f} F1 {f1:.4f}')
results['rf'] = f1
preds_val['rf'] = pv
rf_proba = rf.predict_proba(F_val_s)

# --- Linear SVM (fast baseline) ---
print('  [LinearSVM] training...')
lsvm = LinearSVC(
    C=1.0, class_weight='balanced', dual='auto',
    max_iter=5000, random_state=42,
)
lsvm.fit(F_train_os_s, y_train_os)
pv = lsvm.predict(F_val_s)
acc = accuracy_score(y_val, pv)
f1 = f1_score(y_val, pv, average='macro')
print(f'  [LinearSVM] Val Acc {acc:.4f} F1 {f1:.4f}')
results['lsvm'] = f1
preds_val['lsvm'] = pv

# --- RBF-SVM (subset for tractability, then full with best C) ---
print('  [RBF-SVM] training (full data)...')
rsvm = SVC(
    C=10.0, kernel='rbf', gamma='scale',
    class_weight='balanced', random_state=42,
)
rsvm.fit(F_train_os_s, y_train_os)
pv = rsvm.predict(F_val_s)
acc = accuracy_score(y_val, pv)
f1 = f1_score(y_val, pv, average='macro')
print(f'  [RBF-SVM] Val Acc {acc:.4f} F1 {f1:.4f}')
results['rsvm'] = f1
preds_val['rsvm'] = pv
rsvm_proba = None  # SVC probability is expensive; skip

# ---------------- Summary ----------------
print('\n[5] Classifier comparison (Val Macro F1):')
for name, f1v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {name:12s} {f1v:.4f}')

best_name = max(results, key=results.get)
print(f'\n  Best: {best_name} (F1 {results[best_name]:.4f})')

# ---------------- Test inference ----------------
print('\n[6] Test inference...')
test_dir = os.path.join(DATA_ROOT, 'Test')
result_df = pd.read_csv(TEMPLATE_RESULT)
test_imgs = []
for fname in result_df['id'].values:
    img = cv2.imread(os.path.join(test_dir, str(fname)))
    if img is None:
        test_imgs.append(np.zeros((FEAT_SIZE, FEAT_SIZE, 3), dtype=np.uint8))
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_imgs.append(cv2.resize(img, (FEAT_SIZE, FEAT_SIZE), interpolation=cv2.INTER_AREA))
test_imgs = np.array(test_imgs, dtype=np.uint8)
F_test = extract_features(rn, test_imgs, eval_tf)
F_test_s = scaler.transform(F_test)

# Use best classifier (by val F1) for final prediction
if best_name == 'xgb' and HAS_XGB:
    predictions = xgb.predict(F_test)
elif best_name == 'rf':
    predictions = rf.predict(F_test_s)
elif best_name == 'lsvm':
    predictions = lsvm.predict(F_test_s)
elif best_name == 'rsvm':
    predictions = rsvm.predict(F_test_s)
else:
    predictions = xgb.predict(F_test) if HAS_XGB else rf.predict(F_test_s)

result_df['class'] = predictions.astype(int)
result_df.to_csv(RESULT_PATH, index=False)
print(f'  Saved {len(predictions)} predictions to {RESULT_PATH} (using {best_name})')

# Distribution check vs expected (train/3) -- validation metric ONLY
tc = np.array([np.sum(labels == i) for i in range(NUM_CLASSES)])
exp = tc / 3.0
pred_dist = pd.Series(predictions).value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
diff = pred_dist - exp
print(f'  Dist check: sum|diff|={np.abs(diff).sum():.0f} '
      f'(iter3: 74, iter4: 96), max|diff|={np.abs(diff).max():.0f}, '
      f'min errors >= {np.abs(diff).sum()/2:.0f}')
print('[DONE]')
