"""
Model configurations for the healthcare AI assistant.

Set up your language model with the right parameters for your use case.
"""

from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0.5, timeout=10, max_tokens=1000)
