Write-Host "Starting iter20 (Custom Stem ResNet50 + Full Data)"
conda run -n gpu-torch --no-capture-output python train_resnet_custom_stem.py
Write-Host "iter20 training completed!"
