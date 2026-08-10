Write-Host "Starting iter19 (Hero Model: ConvNeXt + Full Data + SupCon)"
conda run -n gpu-torch --no-capture-output python train_hero.py
Write-Host "iter19 training completed!"

Write-Host "Automatically starting iter20 (Custom Stem ResNet50 + Full Data + SupCon)"
cd C:\hong\python-ws\project-1\iter20-resnet-custom-stem
conda run -n gpu-torch --no-capture-output python train_resnet_custom_stem.py
Write-Host "iter20 training completed!"
