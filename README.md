# IAgricultor

Um assistente especialista em agricultura que roda localmente, utilizando o modelo Qwen2.5 3B em formato GGUF.

## Funcionalidades

- Responde a perguntas sobre agricultura em Português do Brasil.
- Execução local e offline (após o download inicial do modelo).
- Construído em Python com a biblioteca `llama-cpp-python`.

## Como instalar e executar

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/c13neto/IAgricultor.git
   cd IAgricultor
   ```

2. **Instale tudo automaticamente:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   O script cria o ambiente virtual e instala as dependências de `requirements.txt`.

   No Windows PowerShell, execute:
   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   . .\\setup.ps1
   ```

3. **Ative o ambiente virtual:**
   ```bash
   source venv/bin/activate
   ```
   No Windows PowerShell:
   ```powershell
   .\\venv\\Scripts\\Activate.ps1
   ```

   Para sair do ambiente virtual:
   ```bash
   deactivate
   ```

   Se preferir instalar manualmente:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Execute o programa:**
   ```bash
   python main.py
   ```
   *Nota: Na primeira execução, o modelo será baixado automaticamente para o diretório `modelo/`.*

## Como usar

Ao iniciar, um prompt `>` aparecerá. Digite sua pergunta sobre agricultura e pressione Enter.
Para encerrar a conversa, digite `sair`, `tchau`, `exit` ou `quit`.
