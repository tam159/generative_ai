from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

# model = init_chat_model(
#     model="google/gemma-4-31b",  # e.g. "gpt-3.5-turbo" or "lmstudio-llama2"
#     model_provider="openai",  # because LM Studio mimics OpenAI's API
#     base_url="http://localhost:1234/v1",
#     # api_key="not-needed"  # LM Studio accepts any string here
# )


llm = ChatOpenAI(
    model="google/gemma-4-31b",
    # stream_usage=True,
    # temperature=None,
    # max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    # max_retries=2,
    api_key="not-needed",  # If you prefer to pass api key in directly
    base_url="http://127.0.0.1:1234/v1",
    # organization="...",
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)
