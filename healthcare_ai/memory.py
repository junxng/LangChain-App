"""Memory management module.

Add memory to your agent to maintain state across interactions.
This allows the agent to remember previous conversations and context.
"""

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
