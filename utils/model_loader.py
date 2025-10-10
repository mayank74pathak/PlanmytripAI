import os
from dotenv import load_dotenv
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Load .env file
load_dotenv()


class ConfigLoader:
    def __init__(self):
        print("Loaded config...")
        self.config = load_config()

    def __getitem__(self, key):
        return self.config[key]


class ModelLoader(BaseModel):
    model_provider: Literal["groq", "openai"] = "groq"
    config: Optional[ConfigLoader] = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        self.config = ConfigLoader()

    class Config:
        arbitrary_types_allowed = True

    def load_llm(self):
        """Load and return the LLM model based on the provider."""
        print("LLM loading...")
        print(f"Loading model from provider: {self.model_provider}")

        if self.model_provider == "groq":
            print("Loading LLM from Groq...")
            groq_api_key = os.getenv("GROQ_API_KEY")

            # ✅ Safe default model (replace deprecated ones)
            try:
                model_name = self.config["llm"]["groq"]["model_name"]
            except KeyError:
                model_name = "llama-3.3-70b-versatile"  # fallback default

            # Handle deprecated model names gracefully
            if model_name in ["deepseek-r1-distill-llama-70b", "llama-3.1-70b-versatile"]:
                print(f"⚠️  {model_name} is deprecated — using llama-3.3-70b-versatile instead.")
                model_name = "llama-3.3-70b-versatile"

            llm = ChatGroq(model=model_name, api_key=groq_api_key)

        elif self.model_provider == "openai":
            print("Loading LLM from OpenAI...")
            openai_api_key = os.getenv("OPENAI_API_KEY")

            try:
                model_name = self.config["llm"]["openai"]["model_name"]
            except KeyError:
                model_name = "o4-mini"

            llm = ChatOpenAI(model_name=model_name, api_key=openai_api_key)

        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        print(f"✅ LLM Loaded Successfully: {model_name}")
        return llm
