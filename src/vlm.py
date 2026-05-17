from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from collections.abc import Sequence

from openai import APIConnectionError, OpenAI
from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080/v1")
API_KEY = os.getenv("LLAMA_CPP_API_KEY", "local-no-key")
VLM_MODEL = os.getenv("VLM_MODEL", "local-vlm")
VLM_TIMEOUT_SECONDS = float(os.getenv("VLM_TIMEOUT_SECONDS", "75"))


def _prepare_image_for_transport(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA" or image.mode == "LA" or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    if image.mode not in {"RGB", "L"}:
        return image.convert("RGB")
    return image


def _image_to_data_uri(image: "Image.Image | str") -> str:
    if isinstance(image, Image.Image):
        pil_image = _prepare_image_for_transport(image)
    else:
        with Image.open(image) as opened:
            pil_image = _prepare_image_for_transport(opened)

    with BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def ask_vlm(
    prompt: str,
    image: "Image.Image | str | Sequence[Image.Image | str | tuple[str, Image.Image | str]] | None" = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: int = 2048,
) -> str:
    if json_mode:
        prompt = (
            prompt
            + "\n\nIMPORTANT: Return ONLY valid JSON. No preamble, no markdown "
            "code fences, no explanation. Start your response with { or ["
        )

    content = [{"type": "text", "text": prompt}]
    if image is not None:
        images = image if isinstance(image, Sequence) and not isinstance(image, (str, bytes, Image.Image)) else [image]
        for item in images:
            label = None
            image_item = item
            if isinstance(item, tuple):
                label, image_item = item
            if label:
                content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_uri(image_item)}})

    print(
        f"[VLM] server=llama.cpp | model={VLM_MODEL} | "
        f"image={'yes' if image is not None else 'no'} | prompt_len={len(prompt)}"
    )

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=VLM_TIMEOUT_SECONDS)
    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except APIConnectionError:
        print(
            "Is llama-server running? Start it with the Q8_0 llama-server command "
            "from README.md."
        )
        raise

    return response.choices[0].message.content or ""


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_response(text: str) -> dict | list:
    stripped = _strip_markdown_fences(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            break
        else:
            raise

    if isinstance(parsed, (dict, list)):
        return parsed
    raise ValueError("VLM returned valid JSON, but it was not a dict or list.")


def _retry_prompt(original_prompt: str, last_response: str, attempt: int) -> str:
    excerpt = (last_response or "").strip()[:160]
    if attempt <= 1:
        return (
            original_prompt
            + "\n\nYour previous response was not valid JSON. Try again. "
            "Return ONLY JSON. No prose, no markdown fences, no explanation."
        )
    return (
        original_prompt
        + "\n\nLast response (truncated) that could not be parsed as JSON:\n"
        + excerpt
        + "\n\nReturn ONLY a JSON object or array. If you cannot answer, "
        "return an empty JSON object {} with no other text."
    )


def ask_vlm_json(
    prompt: str,
    image=None,
    retries: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dict | list:
    current_prompt = prompt
    last_error: json.JSONDecodeError | None = None

    for attempt in range(retries + 1):
        text = ask_vlm(
            current_prompt,
            image=image,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return _parse_json_response(text)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt < retries:
                current_prompt = _retry_prompt(prompt, text, attempt + 1)
            continue

    raise ValueError(f"VLM did not return valid JSON after {retries + 1} attempts.") from last_error


def _first_diagram() -> Path | None:
    diagrams_dir = Path(__file__).resolve().parents[1] / "data" / "diagrams"
    images = sorted(
        diagrams_dir.glob("*"),
        key=lambda path: (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.name),
    )
    return next((path for path in images if path.suffix.lower() in {".png", ".jpg", ".jpeg"}), None)


if __name__ == "__main__":
    diagram = _first_diagram()
    if diagram is None:
        raise SystemExit("No image found in data/diagrams/")

    print(ask_vlm("Describe this engineering diagram in one sentence.", image=str(diagram)))
    objects = ask_vlm_json(
        "List 3 objects you see. Return a JSON array of strings.",
        image=str(diagram),
    )
    print(objects)
