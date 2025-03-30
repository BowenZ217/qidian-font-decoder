#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to decode and de-obfuscate text from web-based HTML sources
that use randomized font obfuscation for anti-scraping.

It extracts the embedded fonts and CSS rules from the HTML,
generates a character mapping between obfuscated and real fonts using OCR,
then reconstructs the readable paragraphs.

Modules used:
- font_utils: for handling font downloading
- ocr_utils: for OCR recognition and fallback matching
- html_parser: for parsing and extracting content from HTML

Usage example:
    python decode_font.py --html_path chapter.html --chapter_id 1 --save_image --save_dir output/ --use_ocr --use_freq
"""

import argparse
import os

from src import chapter_processor, logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode obfuscated font HTML and reconstruct readable text.")
    parser.add_argument("--html_path", required=True, help="Path to the input HTML file.")
    parser.add_argument("--chapter_id", type=int, required=True, help="Chapter ID used for recursive parsing.")
    parser.add_argument("--save_image", action="store_true", help="Save rendered character images for inspection.")
    parser.add_argument("--save_dir", default="output", help="Directory to save output text and optional images.")
    parser.add_argument("--use_ocr", action="store_true", help="Enable OCR for generating font mapping (fallback matching if not enabled).")
    parser.add_argument("--use_freq", action="store_true", help="Use frequency table to rank image vector")

    args = parser.parse_args()

    log = logger.setup_logging("qidian-decoder")
    os.makedirs(args.save_dir, exist_ok=True)

    chapter_processor.process_chapter(
        html_path=args.html_path,
        chapter_id=args.chapter_id,
        save_image=args.save_image,
        save_dir=args.save_dir,
        use_ocr=args.use_ocr,
        use_freq=args.use_freq
    )
