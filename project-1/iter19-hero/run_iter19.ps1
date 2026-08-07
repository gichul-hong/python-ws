Write-Host "Starting iter19 (Hero Model: ConvNeXt + Full Data + SupCon)"
conda run -n gpu-torch --no-capture-output python train_hero.py
Write-Host "iter19 training completed!"
