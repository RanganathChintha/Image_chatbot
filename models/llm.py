"""Text generation model adapter."""

from groq import Groq

from config import Config


class LLMModel:
    """Generates natural-language answers."""

    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.LLM_MODEL

    def generate(self, prompt: str) -> str:
        """Generate a response from the configured LLM."""
        message = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.choices[0].message.content
