from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-pro-preview-06-05", temperature=0.0)
print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language?"
    )
)
