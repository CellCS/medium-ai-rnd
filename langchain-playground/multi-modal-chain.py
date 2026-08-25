import base64

from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------
# Load image and convert to Base64
# ---------------------------------------------------------------------

IMAGE_PATH = "./assets/ollama_example_img.png"


def image_to_base64(file_path: str) -> str:
    """
    Read an image file and return a Base64-encoded string.
    """
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_b64 = image_to_base64(IMAGE_PATH)


# ---------------------------------------------------------------------
# Initialize LLM
# ---------------------------------------------------------------------

llm = ChatOllama(
    model="qwen3.8:27b",
    temperature=0,
)


# ---------------------------------------------------------------------
# Create multimodal prompt
# ---------------------------------------------------------------------

def create_prompt(data: dict):
    """
    Create a multimodal prompt containing an image and text.
    """
    return [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{data['image']}",
                },
                {
                    "type": "text",
                    "text": data["text"],
                },
            ]
        )
    ]


# ---------------------------------------------------------------------
# Build chain
# ---------------------------------------------------------------------

chain = create_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------
# Run query
# ---------------------------------------------------------------------

response = chain.invoke(
    {
        "text": "What is the download command for the Gemma 7B model?",
        "image": image_b64,
    }
)

print(response)