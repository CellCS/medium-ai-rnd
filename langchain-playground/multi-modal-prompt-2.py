#Describe the image context
#If you already know what kind of image it is, tell the model.

import base64

from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama


IMAGE_PATH = "./assets/ollama_example_img.png"


with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")


llm = ChatOllama(
    model="qwen3.8:27b",
    temperature=0,
)


message = HumanMessage(
    content=[
        {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{image_b64}",
        },
        {
    "type": "text",
    "text": """
The image is a screenshot from an Ollama model page.

Task:
1. Find the section containing model download instructions.
2. Locate the Gemma 7B model.
3. Return only the exact ollama pull command.
Output format:
Return JSON only:
{
  "model_name": "",
  "download_command": "",
  "confidence": ""
}
Do not include explanations.
"""
},
    ]
)

response = llm.invoke([message])

print(response.content)