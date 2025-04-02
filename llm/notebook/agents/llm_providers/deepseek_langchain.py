from dotenv import load_dotenv
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_deepseek import ChatDeepSeek

load_dotenv()

# llm = BaseChatOpenAI(
#     model='deepseek-chat',
#     base_url='https://api.deepseek.com',
#     max_tokens=1024
# )
#
# response = llm.invoke("Hi!")
# print(response.content)



llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    # max_tokens=None,
    # timeout=None,
    # max_retries=2,
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
print(ai_msg.content)
