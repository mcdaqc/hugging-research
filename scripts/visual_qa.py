import base64
import json
import mimetypes
import os
import uuid
from io import BytesIO
from typing import Optional

import requests
from dotenv import load_dotenv
from PIL import Image

from smolagents import Tool, tool


load_dotenv(override=True)


def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def resize_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path


@tool
def visualizer(image_path: str, question: Optional[str] = None) -> str:
    """A tool that can answer questions about attached images.

    Args:
        image_path: The path to the image on which to answer the question. This should be a local path to downloaded image.
        question: The question to answer.
    """
    if not isinstance(image_path, str):
        raise Exception("You should provide at least `image_path` string argument to this tool!")

    add_note = False
    if not question:
        add_note = True
        question = "Please write a detailed caption for this image."

    mime_type, _ = mimetypes.guess_type(image_path)
    base64_image = encode_image(image_path)

    # Configuración para Ollama
    model_id = os.getenv("MODEL_ID", "qwen2.5-coder:3b")
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
    api_key = os.getenv("OPENAI_API_KEY", "ollama")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ],
            }
        ],
        "max_tokens": 1000,
    }

    try:
        response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        output = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        if "Payload Too Large" in str(e):
            new_image_path = resize_image(image_path)
            base64_image = encode_image(new_image_path)
            payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:{mime_type};base64,{base64_image}"
            response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            output = response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error processing image: {str(e)}")

    if add_note:
        output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

    return output
