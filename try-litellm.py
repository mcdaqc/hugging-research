import os

from smolagents import (
    # MANAGED_AGENT_PROMPT,
    CodeAgent,
    HfApiModel,
    LiteLLMModel,
    Model,
    ToolCallingAgent,
)

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

model = LiteLLMModel(
    # "gpt-4o",
    # os.getenv("MODEL_ID", "gpt-4o-mini"),
    os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-V3"),
    custom_role_conversions=custom_role_conversions,
    api_base=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

print(model)

print(model.invoke("Say this is a test"))
