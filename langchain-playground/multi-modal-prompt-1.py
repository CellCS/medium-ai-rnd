#Put instructions into the text prompt

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
            "text":  """
You are analyzing a screenshot.

Instructions:
- Read all visible text carefully.
- Ignore decorative elements.
- If a command is shown in the image, return the exact command.
- Do not guess information that is not visible.
- If the command cannot be found, respond with:
  "Command not found in image."

Question:
What is the download command for the Gemma 7B model?
""",
        },
    ]
)

response = llm.invoke([message])

print(response.content)