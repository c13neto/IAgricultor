import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODELS_DIR = DATA_DIR / "modelo"
DOCS_DIR = DATA_DIR / "documentos"
DATASET_DIR = DATA_DIR / "dataset"
VECTOR_STORE_DIR = DATA_DIR / "chroma_db"

# Ensure essential directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Model constants
REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME_BASE = "qwen2.5-3b-instruct-q4_k_m.gguf"
FILENAME_FT = "qwen2.5-3b-instruct-ft.gguf"

MODELO_BASE_PATH = MODELS_DIR / FILENAME_BASE
MODELO_FT_PATH = MODELS_DIR / FILENAME_FT

# Llama-cpp Hyperparameters
LLM_N_CTX = 2048
LLM_N_BATCH = 512
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_REPEAT_PENALTY = 1.1
LLM_MAX_TOKENS = 512
