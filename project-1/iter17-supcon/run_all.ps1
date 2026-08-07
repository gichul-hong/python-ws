# Run all script
echo Starting 5-Fold ConvNeXt Training (Estimated time: ~4.5 hours)
conda run -n gpu-torch --no-capture-output python -u train_convnext.py 2>&1 | Tee-Object -FilePath run_convnext_5fold.log

echo Starting 5-Fold Scratch CNN Training (Estimated time: ~2.5 hours)
conda run -n gpu-torch --no-capture-output python -u train_scratch.py 2>&1 | Tee-Object -FilePath run_scratch_5fold.log

echo Starting Ensemble
conda run -n gpu-torch --no-capture-output python -u ensemble_5fold.py 2>&1 | Tee-Object -FilePath run_ensemble.log

echo All overnight tasks completed!
