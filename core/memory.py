import tiktoken
from typing import List, Dict, Any
from llama_cpp import ChatCompletionRequestMessage

class MemoryManager:
    """
    Gerencia o histórico de mensagens do chat usando uma janela deslizante (sliding window)
    baseada na contagem de tokens reais, garantindo que o limite de contexto não seja ultrapassado.
    """
    def __init__(self, max_context_tokens: int = 1500, model_encoding: str = "cl100k_base"):
        self.max_context_tokens = max_context_tokens
        self.system_prompt: ChatCompletionRequestMessage | None = None
        self.history: List[ChatCompletionRequestMessage] = []
        try:
            # Utilizamos tiktoken para estimar tokens de maneira rápida e eficiente
            self.encoder = tiktoken.get_encoding(model_encoding)
        except Exception:
            self.encoder = None
        
    def set_system_prompt(self, content: str):
        """Define ou atualiza o system prompt (que nunca é removido da janela)."""
        self.system_prompt = {"role": "system", "content": content}

    def add_message(self, role: str, content: str):
        """Adiciona uma mensagem de usuário ou assistente ao histórico."""
        self.history.append({"role": role, "content": content})
        
    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text) // 4  # Estimativa bruta caso tiktoken falhe
        
    def get_context(self) -> List[ChatCompletionRequestMessage]:
        """
        Retorna a janela de contexto atual (System Prompt + últimas N mensagens que cabem no limite).
        """
        if not self.system_prompt:
            return self.history
            
        context = [self.system_prompt]
        current_tokens = self._estimate_tokens(self.system_prompt["content"])
        
        messages_to_include = []
        # Percorre o histórico de trás para frente para pegar as mais recentes
        for msg in reversed(self.history):
            msg_tokens = self._estimate_tokens(str(msg.get("content", "")))
            
            if current_tokens + msg_tokens > self.max_context_tokens:
                break
                
            messages_to_include.insert(0, msg)
            current_tokens += msg_tokens
            
        context.extend(messages_to_include)
        return context
