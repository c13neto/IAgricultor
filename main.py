import sys
from utils.downloader import download_model
from config import MODELO_BASE_PATH, MODELO_FT_PATH
from core.llm_service import LLMService
from core.memory import MemoryManager
from core.rag_engine import RAGEngine
from evaluation.evaluator import Evaluator

def iniciar_chat(caminho_modelo) -> None:
    try:
        llm_service = LLMService(caminho_modelo)
    except FileNotFoundError as e:
        print(e)
        return

    memory = MemoryManager()
    memory.set_system_prompt(
        "Você é o IAgricultor, um assistente especialista em agricultura. "
        "Responda em Português do Brasil de forma clara e objetiva."
    )
    
    print("\nIAgricultor pronto! (digite 'sair' para voltar ao menu)\n")
    
    while True:
        mensagem_input = input("> ")
        if mensagem_input.lower().strip() in ["sair", "tchau", "exit", "quit"]:
            break
            
        if not mensagem_input.strip():
            continue
            
        memory.add_message("user", mensagem_input)
        contexto_atual = memory.get_context()
        
        resposta = llm_service.generate_chat_response(contexto_atual)
        memory.add_message("assistant", resposta)
        
        print(f"\n{resposta}\n")

def executar_avaliacao() -> None:
    try:
        llm = LLMService(MODELO_BASE_PATH)
    except FileNotFoundError as e:
        print(e)
        return
        
    rag = RAGEngine()
    evaluator = Evaluator(llm, rag)
    evaluator.run_benchmark()

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
            iniciar_chat(MODELO_FT_PATH)
        elif opcao == "2":
            iniciar_chat(MODELO_BASE_PATH)
        elif opcao == "3":
            executar_avaliacao()
        elif opcao == "4":
            print("Saindo...")
            sys.exit(0)
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    menu()