"""Text generation model adapter."""

import logging

from groq import Groq

from config import Config

logger = logging.getLogger(__name__)


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
        logger.info("LLM generated response successfully")
        return message.choices[0].message.content
