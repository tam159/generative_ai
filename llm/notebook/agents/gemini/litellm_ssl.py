import os
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

# Set SSL certificate for LiteLLM/httpx
os.environ["SSL_CERT_FILE"] = "/Users/may/j/netskope.pem"


response = completion(
    model="gemini/gemini-2.5-flash",
    messages=[{"role": "user", "content": "Explain how AI works in a few words"}]
)

print(response.choices[0].message.content)