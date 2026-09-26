python -m venv venv
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Instalação concluída. Para ativar o ambiente nesta sessão, execute:"
Write-Host ". .\venv\Scripts\Activate.ps1"
Write-Host "Depois, execute: python main.py"