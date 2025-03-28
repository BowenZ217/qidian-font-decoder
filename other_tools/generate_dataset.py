#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate character images from a font file.

This script renders Chinese characters (and optionally Latin, digits, etc.) into grayscale images
using a specified font. Useful for generating training data for OCR or classification models.

Usage:
    python tools/generate_dataset.py \
        --font fonts/SourceHanSansSC-Regular.woff2 \
        --chars data/common_chars.txt \
        --output data/images \
        --image_size 224 \
        --font_size 160
"""

import os
import argparse
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont  # Not directly used, but may be helpful in extensions


def load_characters(char_file):
    """Load characters from a file, one character per line."""
    with open(char_file, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f if line.strip()]
    return chars


def generate_char_image(char, font, image_size=64, font_size=48):
    """
    Render a single character to a grayscale image.

    Args:
        char (str): The character to render.
        font (ImageFont.FreeTypeFont): PIL font object.
        image_size (int): Output image dimensions (square).
        font_size (int): Font size used for rendering.

    Returns:
        PIL.Image: Rendered image.
    """
    img = Image.new("L", (image_size, image_size), color=255)  # white background
    draw = ImageDraw.Draw(img)

    # Compute bounding box using textbbox (compatible with Pillow >= 10)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (image_size - w) // 2 - bbox[0]
    y = (image_size - h) // 2 - bbox[1]

    draw.text((x, y), char, fill=0, font=font)  # black text
    return img


def generate_dataset(font_path, char_file, output_dir, image_size=64, font_size=48):
    """
    Generate a dataset of character images from a font.

    Args:
        font_path (str): Path to font file (.ttf, .woff2, etc).
        char_file (str): Path to character list file.
        output_dir (str): Directory to save output images and labels.txt.
        image_size (int): Size of each output image.
        font_size (int): Font size used for rendering.
    """
    os.makedirs(output_dir, exist_ok=True)
    chars = load_characters(char_file)

    num_digits = max(5, len(str(len(chars))))  # e.g. 00001.png
    label_file = os.path.join(output_dir, "labels.txt")

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"[X] Failed to load font: {e}")
        return

    with open(label_file, 'w', encoding='utf-8') as label_f:
        for idx, char in enumerate(chars):
            img = generate_char_image(char, font, image_size, font_size)
            file_id = f"{idx:0{num_digits}d}"
            img.save(os.path.join(output_dir, f"{file_id}.png"))
            label_f.write(f"{file_id}\t{char}\n")

    print(f"[✓] Generated {len(chars)} images in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate character images from a font file."
    )
    parser.add_argument("--font", required=True, help="Path to font file (.ttf, .woff2, etc)")
    parser.add_argument("--chars", required=True, help="Path to file containing characters (one per line)")
    parser.add_argument("--output", required=True, help="Output directory to save images and labels")
    parser.add_argument("--image_size", type=int, default=224, help="Output image size (square)")
    parser.add_argument("--font_size", type=int, default=160, help="Font size used for rendering")

    args = parser.parse_args()

    generate_dataset(
        font_path=args.font,
        char_file=args.chars,
        output_dir=args.output,
        image_size=args.image_size,
        font_size=args.font_size
    )
