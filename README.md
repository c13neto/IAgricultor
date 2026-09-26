# IAgricultor

Um assistente especialista em agricultura que roda localmente, utilizando o modelo DeepSeek-R1-Distill-Qwen-1.5B-GGUF.

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

2. **Crie um ambiente virtual (recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows, use: venv\Scripts\activate
   ```

3. **Instale as dependências necessárias:**
   ```bash
   pip install llama-cpp-python huggingface-hub python-dotenv
   ```

4. **Execute o programa:**
   ```bash
   python main.py
   ```
   *Nota: Na primeira execução, o modelo será baixado automaticamente para o diretório `modelo/`.*

## Como usar

Ao iniciar, um prompt `>` aparecerá. Digite sua pergunta sobre agricultura e pressione Enter.
Para encerrar a conversa, digite `sair`, `tchau`, `exit` ou `quit`.
