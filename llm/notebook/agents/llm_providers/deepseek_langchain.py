from dotenv import load_dotenv
from langchain_openai.chat_models.base import BaseChatOpenAI

load_dotenv()

llm = BaseChatOpenAI(
    model='deepseek-chat',
    base_url='https://api.deepseek.com',
    max_tokens=1024
)

response = llm.invoke("Hi!")
print(response.content)
