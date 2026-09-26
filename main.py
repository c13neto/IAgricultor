import re
from pathlib import Path
from typing import cast
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

caminho_modelo = Path("modelo/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf")


def download_model() -> None:
    load_dotenv()
    if not caminho_modelo.is_file():
        hf_hub_download(
            repo_id="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            filename="DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
            local_dir="modelo",
        )


def extrair_resposta(texto_bruto: str) -> str:

    texto_limpo = re.sub(r"<think>.*?</think>", "", texto_bruto, flags=re.DOTALL)

    texto_limpo = re.sub(r"<think>.*", "", texto_limpo, flags=re.DOTALL)
    
    return texto_limpo.strip()


def mensagem_user(mensagem: str, llm: Llama) -> str:
    prompt = (
        "<｜User｜>Você é um assistente especialista em agricultura. "
        "Responda à pergunta a seguir exclusivamente em Português do Brasil, de forma clara, "
        "correta e objetiva.\n\n"
        f"Pergunta: {mensagem}<｜Assistant｜>"
    )

    output = cast(
        dict,
        llm(
            prompt,
            max_tokens=1024,
            temperature=0.3,
            top_p=0.85,
            stop=["<｜User｜>", "<｜end of sentence｜>", "</s>"],
        ),
    )
    
    texto_gerado = str(output["choices"][0]["text"])
    return extrair_resposta(texto_gerado)


def main() -> None:
    download_model()
    
    llm = Llama(model_path=str(caminho_modelo), n_ctx=2048, verbose=False)

    while True:
        mensagem_input = input("> ")
        if mensagem_input.lower().strip() in ["sair", "tchau", "exit", "quit"]:
            print("Até logo!")
            break

        if not mensagem_input.strip():
            continue

        mensagem_output = mensagem_user(mensagem_input, llm)
        print(f"\n{mensagem_output}\n")


if __name__ == "__main__":
    main()