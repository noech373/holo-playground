"""Ask a Holo model, served by llama-server, where to click on a screenshot."""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from holo_playground.screen import COORD_SCALE, Screenshot

# Grounding prompt from https://hub.hcompany.ai/models-api/element-localization.md
PROMPT = """Localize an element on the GUI image according to the provided target and output a click position.
 * You must output a valid JSON following the format: {schema}
Your target is:
{element}"""

class Click(BaseModel):
    x: int = Field(ge=0, le=COORD_SCALE, description="X coordinate as integer in [0, 1000]")
    y: int = Field(ge=0, le=COORD_SCALE, description="Y coordinate as integer in [0, 1000]")


CLICK_SCHEMA = Click.model_json_schema()


@dataclass
class Prediction:
    click: Click
    wall_s: float
    timings: dict = field(default_factory=dict)  # llama-server per-request timings


def locate(client: OpenAI, screenshot: Screenshot, element: str) -> Prediction:
    """Return the click predicted for `element`, constrained to valid JSON by the server."""
    prompt = PROMPT.format(schema=json.dumps(CLICK_SCHEMA), element=element)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model="holo",  # llama-server serves a single model and ignores this name
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": screenshot.data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.0,
        response_format={"type": "json_schema", "json_schema": {"name": "click", "schema": CLICK_SCHEMA}},
    )
    wall_s = time.perf_counter() - start
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("empty response from the model")
    timings = (response.model_extra or {}).get("timings", {})
    return Prediction(Click.model_validate_json(content), wall_s, timings)


def draw_clicks(
    screenshot: Screenshot,
    clicks: list[tuple[float, float]],
    out: Path,
    boxes: list[list[int]] | None = None,
) -> None:
    """Save a copy of the screenshot with ground-truth boxes (green) and clicks (red)."""
    with Image.open(screenshot.path) as img:
        annotated = img.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for box in boxes or []:
        draw.rectangle(box, outline="lime", width=2)
    r = 8
    for x, y in clicks:
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=3)
    annotated.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="screenshot to search")
    parser.add_argument("element", help="natural-language description of the element to click")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    args = parser.parse_args()

    screenshot = Screenshot.load(args.image)
    client = OpenAI(base_url=args.base_url, api_key="none")
    pred = locate(client, screenshot, args.element)

    x_px, y_px = screenshot.to_pixels(pred.click.x, pred.click.y)
    print(
        f"image {screenshot.width}x{screenshot.height}"
        f" | model ({pred.click.x}, {pred.click.y}) | pixels ({x_px:.0f}, {y_px:.0f})"
    )
    t = pred.timings
    if t:
        print(
            f"prefill {t['prompt_n']} tok in {t['prompt_ms'] / 1000:.1f} s"
            f" | gen {t['predicted_n']} tok in {t['predicted_ms'] / 1000:.1f} s"
            f" | total {pred.wall_s:.1f} s"
        )
    out = args.image.with_name(f"{args.image.stem}_click.png")
    draw_clicks(screenshot, [(x_px, y_px)], out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
