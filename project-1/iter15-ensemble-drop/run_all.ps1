# Run all script for iter15 (2 Heavy + 3 Light Ensemble, 3-Fold CV)
echo "Starting 3-Fold ConvNeXt Training (Heavy 1)"
conda run -n gpu-torch --no-capture-output python -u train_convnext.py 2>&1 | Tee-Object -FilePath run_convnext_3fold.log

echo "Starting 3-Fold EfficientNet Training (Heavy 2)"
conda run -n gpu-torch --no-capture-output python -u train_efficientnet.py 2>&1 | Tee-Object -FilePath run_efficientnet_3fold.log

echo "Starting 3-Fold Scratch CNN Training (Light 1)"
conda run -n gpu-torch --no-capture-output python -u train_scratch.py 2>&1 | Tee-Object -FilePath run_scratch_3fold.log

echo "Starting 3-Fold ResNet Training (Light 2)"
conda run -n gpu-torch --no-capture-output python -u train_resnet.py 2>&1 | Tee-Object -FilePath run_resnet_3fold.log

echo "Starting 3-Fold MobileNet Training (Light 3)"
conda run -n gpu-torch --no-capture-output python -u train_mobilenet.py 2>&1 | Tee-Object -FilePath run_mobilenet_3fold.log

echo "Starting 5-Model Ensemble Optimization"
conda run -n gpu-torch --no-capture-output python -u ensemble_3fold.py 2>&1 | Tee-Object -FilePath run_ensemble.log

echo "All 2+3 Ensemble tasks completed!"
