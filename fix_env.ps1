# Fix Python environment issues
# Run this with: powershell -ExecutionPolicy Bypass -File fix_env.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fixing Python Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Activate venv
& .\.venv\Scripts\Activate.ps1

Write-Host "`n[1/4] Uninstalling bitsandbytes and flash-attn..." -ForegroundColor Yellow
pip uninstall bitsandbytes flash-attn -y --quiet

Write-Host "[2/4] Downgrading NumPy to v1.x..." -ForegroundColor Yellow
pip install "numpy==1.24.3" --upgrade --force-reinstall --quiet

Write-Host "[3/4] Installing compatible PyTorch..." -ForegroundColor Yellow
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

Write-Host "[4/4] Installing remaining requirements..." -ForegroundColor Yellow
pip install scipy transformers peft -q

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Environment fixed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nTesting imports..." -ForegroundColor Cyan
python -c "import torch; import transformers; import peft; import numpy; print('[OK] All imports successful'); print(f'NumPy: {numpy.__version__}'); print(f'PyTorch: {torch.__version__}'); print(f'Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')" 

Write-Host "`nYou can now run: python lab_pipeline.py" -ForegroundColor Green
