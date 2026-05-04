import base64
import requests
from fastapi import APIRouter, UploadFile, File, Form
from .config import OLLAMA_BASE_URL, LLM_MODEL

router = APIRouter()


def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def call_ollama_vision(prompt: str, image_b64: str, model: str):
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }
        ],
        "stream": False
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


@router.post("/vision")
async def vision_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form(None)
    ):
    image_bytes = await file.read()
    image_b64 = encode_image(image_bytes)

    user_prompt = prompt.strip() if prompt else ""

    # 🧠 FORCE STRUCTURED OUTPUT FORMAT
    system_prompt = """
    You are a vision AI.

    You MUST always respond in valid JSON with this structure:

    {
    "summary": "",
    "objects": [],
    "scene": "",
    "text_in_image": "",
    "user_question_answer": "",
    "safety_notes": ""
    }

    Rules:
    - Always return JSON only
    - No explanations
    - No markdown
    - No extra text
    """

    # 🧠 merge user intent into structured instruction
    final_prompt = f"""
    {system_prompt}

    User request:
    {user_prompt if user_prompt else "Describe the image in detail."}
    """

    result = call_ollama_vision(final_prompt, image_b64, LLM_MODEL)

    return {
        "type": "vision_result",
        "data": result["message"]["content"]
    }