#Add a system prompt
#This is often better because it separates behavior instructions from the actual question.


import base64
from langchain.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

IMAGE_PATH = "./assets/ollama_example_img.png"


with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")


llm = ChatOllama(
    model="qwen3.8:27b",
    temperature=0,
)


messages = [
    SystemMessage(
        content="""
You are an expert at reading screenshots and technical documentation.

Rules:
- Extract information only from the image.
- Do not hallucinate missing values.
- Return concise answers.
- Preserve commands exactly as shown.
"""
    ),
    HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{image_b64}",
            },
            {
                "type": "text",
                "text": "What is the download command for the Gemma 7B model?",
            },
        ]
    ),
]

response = llm.invoke(messages)
print(response.content)