#!/usr/bin/env bash
set -e

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "Instalação concluída. Para ativar o ambiente nesta sessão, execute:"
echo "source venv/bin/activate"
echo "Depois, execute: python main.py"