"""LangChain agent application."""

import logging

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from healthcare_ai.config import model
from healthcare_ai.memory import checkpointer
from healthcare_ai.prompt import system_prompt
from healthcare_ai.response import ResponseFormat
from healthcare_ai.tools import (
    Context,
    get_user_location,
    get_weather_for_location,
)

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)
# `thread_id` is a unique identifier for a given conversation.
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1"),
)

logger.info(response["structured_response"])
ResponseFormat(
    punny_response=(
        "Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long!"
        "I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, "
        "I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!"
    ),
    weather_conditions="It's always sunny in Florida!",
)


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]}, config=config, context=Context(user_id="1")
)

logger.info(response["structured_response"])
ResponseFormat(
    punny_response=(
        "You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. "
        "I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. "
        "Have a 'sun-sational' day in the Florida sunshine!"
    ),
    weather_conditions=None,
)
