#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a list of characters (e.g., Chinese + alphanumerics) and their image vectors from a font.

This script renders each character into a square grayscale image using the specified font,
resizes it to a smaller vector (e.g., 32x32), flattens it, and saves the result to a .npy file.
It also saves the character list to a .txt file, line by line.

Usage:
    python other_tools/generate_char_vectors.py \
        --font fonts/SourceHanSansSC-Regular.woff2 \
        --start 0x4E00 \
        --end 0x9FA5 \
        --extra "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
        --output data/char_vectors.npy \
        --image_size 64 \
        --resize 32 \
        --font_size 48
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def generate_char_vectors(font_path, start_idx, end_idx, extra_chars, output_file,
                          canvas_size=64, vector_size=(32, 32), font_size=48, save_txt=True):
    font = ImageFont.truetype(font_path, font_size)

    # Collect all characters
    unicode_chars = [chr(i) for i in range(start_idx, end_idx + 1)]
    all_chars = sorted(set(unicode_chars + list(extra_chars)))

    vectors = []
    valid_chars = []

    for ch in tqdm(all_chars, desc="Generating vectors"):
        try:
            # Create white canvas
            img = Image.new("L", (canvas_size, canvas_size), color=255)
            draw = ImageDraw.Draw(img)

            bbox = draw.textbbox((0, 0), ch, font=font)
            if bbox is None:
                continue
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (canvas_size - w) // 2 - bbox[0]
            y = (canvas_size - h) // 2 - bbox[1]
            draw.text((x, y), ch, fill=0, font=font)

            # Resize and flatten
            img_resized = img.resize(vector_size)
            vec = np.array(img_resized).astype(np.float32).flatten() / 255.0
            vectors.append(vec)
            valid_chars.append(ch)
        except Exception as e:
            print(f"[!] Skipping char '{ch}' due to error: {e}")
            continue

    # vectors = np.array(vectors)
    vectors = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    vectors /= norms
    
    np.save(output_file, vectors)
    print(f"[✓] Saved vectors to: {output_file} (shape: {vectors.shape})")

    if save_txt:
        txt_path = os.path.splitext(output_file)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for ch in valid_chars:
                f.write(ch + "\n")
        print(f"[✓] Saved character list to: {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate character image vectors from font.")
    parser.add_argument("--font", required=True, help="Path to the font file (.ttf/.otf/.woff2)")
    parser.add_argument("--start", type=lambda x: int(x, 0), required=True, help="Unicode start index (e.g., 0x4E00)")
    parser.add_argument("--end", type=lambda x: int(x, 0), required=True, help="Unicode end index (e.g., 0x9FA5)")
    parser.add_argument("--extra", type=str, default="", help="Extra characters to include (e.g., 'abc123')")
    parser.add_argument("--output", required=True, help="Path to output .npy file")
    parser.add_argument("--image_size", type=int, default=64, help="Canvas size (default: 64)")
    parser.add_argument("--resize", type=int, default=32, help="Resize image to this square size before flattening")
    parser.add_argument("--font_size", type=int, default=48, help="Font size used for rendering")
    args = parser.parse_args()

    generate_char_vectors(
        font_path=args.font,
        start_idx=args.start,
        end_idx=args.end,
        extra_chars=args.extra,
        output_file=args.output,
        canvas_size=args.image_size,
        vector_size=(args.resize, args.resize),
        font_size=args.font_size,
        save_txt=True
    )
