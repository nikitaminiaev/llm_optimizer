import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from local_llm.ollamaClient import OllamaClient


class CreatorBotWrapper:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:14b"):
        self.ollama_client = OllamaClient(base_url, model)

    def create_text(self, prompt: str, stream: bool = False, max_tokens: int = 100, temp: float = 0.5) -> str:
        options = {
            "temperature": temp,
        }
        return self.ollama_client._make_ollama_request(prompt, options)

    def filtration(self, text: str) -> str:
        # Здесь можно добавить дополнительную фильтрацию текста, если необходимо
        return text