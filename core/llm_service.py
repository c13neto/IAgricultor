import os
from pathlib import Path
from typing import cast
from llama_cpp import Llama, ChatCompletionRequestMessage
from config import LLM_N_CTX, LLM_N_BATCH

class LLMService:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"[-] Modelo não encontrado em: {model_path}")
        
        num_threads = min(os.cpu_count() or 4, 6)
        print(f"[+] Carregando LLM de {model_path.name} usando {num_threads} threads...")
        
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=LLM_N_CTX,
            n_batch=LLM_N_BATCH,
            n_threads=num_threads,
            n_gpu_layers=0,
            verbose=False
        )

    def generate_chat_response(self, messages: list[ChatCompletionRequestMessage], 
                               max_tokens: int = 512, 
                               temperature: float = 0.7,
                               top_p: float = 0.9,
                               repeat_penalty: float = 1.1) -> str:
        
        output = cast(
            dict,
            self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            ),
        )

        return str(output["choices"][0]["message"]["content"]).strip()
        
    def generate_text(self, prompt: str, max_tokens: int = 128, temperature: float = 0.2) -> str:
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["\nPergunta:", "\nUsuário:", "###", "Exemplo:"],
            echo=False,
            temperature=temperature
        )
        return response["choices"][0]["text"].strip()
