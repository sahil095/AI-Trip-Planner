import os
from dotenv import load_dotenv
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

class ConfigLoader:

    def __init__(self):
        print(f"Loaded Config...")
        self.cofig = load_config()

    def __getitem__(self, key):
        return self.cofig[key]


class ModelLoader(BaseModel):

    model_provider: Literal["groq", "openai"] = "groq"
    config: Optional[ConfigLoader] = Field(default=None, exclude=True)
    
    def model_post_init(self, __context:Any) -> None:
        self.config = ConfigLoader()

    class Config:
        arbitary_types_allowed = True

    
    def load_ll(self):
        """
        Load and return the LLM Model
        """

        print("LLM loading...")
        print(f"Loading model from provider: {self.model_provider}")

        if self.model_provider == "groq":
            print("Loading LLM from groq.........")
            groq_api_key = os.getenv("GROQ_API_KEY")
            model_name = self.config["llm"]["groq"]["model_name"]
            llm = ChatGroq(model=model_name, api_key=groq_api_key)
        elif self.model_provider == "openai":
            print("Loading LLM from OpenAI................")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            model_name = self.config['llm']['openai']['model_name']
            llm = ChatOpenAI(model_name=model_name, api_key=openai_api_key)
        
        return llm

