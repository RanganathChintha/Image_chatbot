# llm_model.py
from groq import Groq
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.LLM_MODEL
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        try:
            message = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            logger.info("LLM generated response successfully")
            return message.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise