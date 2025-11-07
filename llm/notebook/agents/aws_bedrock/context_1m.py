from dotenv import load_dotenv
import boto3

load_dotenv()

# Configure Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
    # region_name='ap-northeast-1'
)

def converse_claude_sonnet4_1m(prompt_text):
    response = bedrock.converse(
        # modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
        modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        # modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
        # modelId="apac.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt_text}]
            }
        ],
        inferenceConfig={
            "maxTokens": 4000,
        },
        additionalModelRequestFields={
            "anthropic_beta": ["context-1m-2025-08-07"]  # Header for 1M context
        }
    )

    return response['output']['message']['content'][0]['text']

# print(converse_claude_sonnet4_1m("What is the meaning of life?"))

# Example usage (this will cause 213K input tokens to be sent to Claude, at a cost of ~$0.65)
with open('on_the_origin_of_species.txt', 'r', encoding='utf-8') as file:
    large_document = file.read()
result = converse_claude_sonnet4_1m(f"Provide a numbered list of Darwin's main arguments about natural selection, without further explanation: {large_document}")
print(result)