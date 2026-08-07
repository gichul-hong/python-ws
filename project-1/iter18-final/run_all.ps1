Write-Host "Starting iter18 (4-Model Pruned+SupCon Ensemble)"

conda run -n gpu-torch --no-capture-output python train_convnext.py
conda run -n gpu-torch --no-capture-output python train_resnet.py
conda run -n gpu-torch --no-capture-output python train_efficientnet.py
conda run -n gpu-torch --no-capture-output python train_mobilenet.py

Write-Host "Ensemble"
conda run -n gpu-torch --no-capture-output python ensemble_4models.py

Write-Host "All overnight tasks completed!"
