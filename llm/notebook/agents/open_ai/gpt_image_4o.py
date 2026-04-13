import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

# SSL certificate for corporate proxy (Netskope)
SSL_CERT_PATH = os.environ.get("SSL_CERT_FILE", "/Users/may/jitera/netskope.pem")

# Responses API endpoint (supports image generation with gpt-4o)
url = "https://api.openai.com/v1/responses"
headers = {
    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    "Content-Type": "application/json"
}

# Input image path
input_image_path = "dark-mode-custom-agent.png"

# Read and encode image
with open(input_image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "gpt-4o",
    "input": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_b64}"
                },
                {
                    "type": "input_text",
                    "text": "Edit this image: place a flower at the center"
                }
            ]
        }
    ],
    "tools": [
        {
            "type": "image_generation",
            "quality": "low",
            "size": "1024x1024"
        }
    ]
}

# response = requests.post(url, headers=headers, json=payload, verify=SSL_CERT_PATH)
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    result = response.json()
    # Extract image from response output
    for output in result.get("output", []):
        if output.get("type") == "image_generation_call":
            image_b64 = output.get("result")
            if image_b64:
                image_bytes = base64.b64decode(image_b64)
                output_path = "output_image.png"
                with open(output_path, "wb") as out_file:
                    out_file.write(image_bytes)
                print(f"Image saved to {output_path}")
                break
    else:
        print("No image generation output found")
        print(result)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
