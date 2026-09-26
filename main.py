import sys
from pathlib import Path
from typing import cast
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import ChatCompletionRequestMessage, Llama
from avaliador import executar_avaliacao

REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME_BASE = "qwen2.5-3b-instruct-q4_k_m.gguf"
FILENAME_FT = "qwen2.5-3b-instruct-ft.gguf"

MODELO_BASE = Path(f"modelo/{FILENAME_BASE}")
MODELO_FT = Path(f"modelo/{FILENAME_FT}")


def download_model() -> None:
    load_dotenv()
    if not MODELO_BASE.is_file():
        print("Baixando o modelo base...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME_BASE,
            local_dir="modelo",
        )


def resumir_mensagens_antigas(
    mensagens: list[ChatCompletionRequestMessage], llm: Llama
) -> str:
    texto_conversas = "\n".join(
        [f"{m['role'].upper()}: {m.get('content', '')}" for m in mensagens]
    )

    prompt_resumo: list[ChatCompletionRequestMessage] = [
        {
            "role": "system",
            "content": (
                "Você é um assistente encarregado de resumir histórico de conversas. "
                "Crie um resumo muito curto, objetivo e em tópicos em Português do Brasil, "
                "destacando apenas dados do usuário e dúvidas agrícolas citadas."
            ),
        },
        {
            "role": "user",
            "content": f"Resuma o seguinte histórico de conversa de forma concisa:\n\n{texto_conversas}",
        },
    ]

    output = cast(
        dict,
        llm.create_chat_completion(
            messages=prompt_resumo,
            max_tokens=200,
            temperature=0.3,
        ),
    )

    return str(output["choices"][0]["message"]["content"]).strip()


def compactar_historico_se_necessario(
    historico: list[ChatCompletionRequestMessage], llm: Llama
) -> list[ChatCompletionRequestMessage]:
    LIMITE_DISPARO = 15

    if len(historico) > LIMITE_DISPARO:
        print("\n[Sistema: Resumindo trecho antigo do histórico para poupar memória...]\n")
        mensagens_antigas = historico[1:-10]
        resumo_texto = resumir_mensagens_antigas(mensagens_antigas, llm)

        mensagem_resumo: ChatCompletionRequestMessage = {
            "role": "system",
            "content": f"Resumo do contexto anterior da conversa:\n{resumo_texto}",
        }

        historico = [historico[0], mensagem_resumo] + historico[-10:]

    return historico


def mensagem_user(historico: list[ChatCompletionRequestMessage], llm: Llama) -> str:
    mensagens_para_enviar = [historico[0]] + historico[-10:]

    output = cast(
        dict,
        llm.create_chat_completion(
            messages=mensagens_para_enviar,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
        ),
    )

    return str(output["choices"][0]["message"]["content"]).strip()


def iniciar_chat(caminho_modelo: Path) -> None:
    if not caminho_modelo.is_file():
        print(f"\n[-] Modelo não encontrado em: {caminho_modelo}")
        return

    print(f"\nCarregando o modelo: {caminho_modelo.name}...")
    llm = Llama(
        model_path=str(caminho_modelo),
        n_ctx=2048,
        n_batch=512,
        n_gpu_layers=0,
        verbose=False
    )

    historico: list[ChatCompletionRequestMessage] = [
        {
            "role": "system",
            "content": (
                "Você é o IAgricultor, um assistente especialista em agricultura. "
                "Responda em Português do Brasil de forma clara e objetiva."
            ),
        }
    ]

    print("IAgricultor pronto! (digite 'sair' para voltar ao menu)\n")

    while True:
        mensagem_input = input("> ")
        if mensagem_input.lower().strip() in ["sair", "tchau", "exit", "quit"]:
            break

        if not mensagem_input.strip():
            continue

        historico.append({"role": "user", "content": mensagem_input})
        historico = compactar_historico_se_necessario(historico, llm)
        resposta = mensagem_user(historico, llm)
        historico.append({"role": "assistant", "content": resposta})

        print(f"\n{resposta}\n")


def menu() -> None:
    download_model()

    while True:
        print("\n=== MENU IAGRICULTOR ===")
        print("1 - Rodar modelo Fine-Tuned")
        print("2 - Rodar modelo Normal (Base)")
        print("3 - Executar Avaliação (Zero-Shot, One-Shot, Few-Shot e RAG)")
        print("4 - Sair")

        opcao = input("\nEscolha uma opção: ").strip()

        if opcao == "1":
            iniciar_chat(MODELO_FT)
        elif opcao == "2":
            iniciar_chat(MODELO_BASE)
        elif opcao == "3":
            executar_avaliacao()
        elif opcao == "4":
            print("Saindo...")
            sys.exit(0)
        else:
            print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    menu()