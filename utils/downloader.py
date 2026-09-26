from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from config import MODELS_DIR, REPO_ID, FILENAME_BASE, MODELO_BASE_PATH

def download_model() -> None:
    """Downloads the base LLM model from Hugging Face if it's not already downloaded."""
    load_dotenv()
    if not MODELO_BASE_PATH.is_file():
        print(f"[+] Baixando o modelo base para {MODELS_DIR}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME_BASE,
            local_dir=str(MODELS_DIR),
        )
        print("[+] Download concluído.")
    else:
        print("[+] Modelo base já encontrado localmente.")
