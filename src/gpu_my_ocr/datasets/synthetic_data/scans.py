"""Deterministic grayscale scans with explicit, bounded visual degradation."""

import io
import random
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageChops, ImageFilter

DPI = 150
BLUR_RADIUS = 0.45
NOISE_AMPLITUDE = 8


def transformations(variant: str, seed: int) -> dict[str, Any]:
    """Return versioned rendering parameters for a variant and document noise seed."""
    if variant == "digital":
        return {"type": "selectable_text"}
    result = {
        "type": "raster_only",
        "dpi": DPI,
        "colorspace": "grayscale",
        "encoding": "lossless_png",
    }
    if variant == "scan_degraded":
        result.update(
            {
                "blur_radius_px": BLUR_RADIUS,
                "noise_seed": seed,
                "noise_amplitude": NOISE_AMPLITUDE,
                "noise_algorithm": "Random.randbytes modulo 17 minus 8; clipped addition",
            }
        )
    elif variant != "scan_high":
        raise ValueError(f"Unknown PDF variant: {variant}")
    return result


def degrade_image(image: Image.Image, seed: int) -> Image.Image:
    """Return blurred grayscale pixels plus seeded bounded uniform noise.

    Noise is generated via random.Random(seed).randbytes, mapped modulo 17 to
    [-8, 8], then added with clipping. Modulo introduces a tiny documented bias.
    """
    blurred = image.convert("L").filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    noise = Image.frombytes(
        "L", image.size, random.Random(seed).randbytes(image.width * image.height)
    )
    noise = noise.point(
        [128 + value % (2 * NOISE_AMPLITUDE + 1) - NOISE_AMPLITUDE for value in range(256)]
    )
    return ImageChops.add(blurred, noise, offset=-128)


def write_image_pdf(image: Image.Image, output: Path, width: float, height: float) -> None:
    """Insert raster pixels only into a deterministic PDF matching original page dimensions."""
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=width, height=height)
        page.insert_image(page.rect, stream=stream.getvalue())
        pdf.save(output, deflate=True, no_new_id=True)


def render_scans(digital: Path, high: Path, degraded: Path, seed: int) -> None:
    """Rasterize one digital page at 150 DPI and write high/degraded image-only PDFs."""
    with pymupdf.open(digital) as pdf:
        page = pdf[0]
        pixmap = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY, alpha=False)
        image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
        width, height = page.rect.width, page.rect.height
    write_image_pdf(image, high, width, height)
    write_image_pdf(degrade_image(image, seed), degraded, width, height)
