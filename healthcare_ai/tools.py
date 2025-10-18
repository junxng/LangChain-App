"""
Tools for the healthcare AI assistant.

Tools let a model interact with external systems by calling functions you define. Tools can depend on runtime
context and also interact with agent memory.

It should be well-documented: their name, description, and argument names become part of the model's prompt.
LangChain's @tool decorator adds metadata and enables runtime injection via the ToolRuntime parameter.
"""

from dataclasses import dataclass

from langchain.tools import ToolRuntime, tool


@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@dataclass
class Context:
    """Custom runtime context schema."""

    user_id: str


@tool
def get_user_location(runtime: ToolRuntime[Context, str]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
