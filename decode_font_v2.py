#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to decode and de-obfuscate text from web-based HTML sources
that use randomized font obfuscation for anti-scraping.

Modules used:
- font_utils: for handling font downloading
- ocr_utils: for OCR recognition and fallback matching
- html_parser: for parsing and extracting content from HTML

Usage example:
    python decode_font_v2.py --html_folder data/html --save_image --save_dir output/ --use_ocr --use_freq
"""

import argparse
import os

from src import chapter_processor, logger

# ----------------------
# Main process
# ----------------------

def main(args):
    html_folder = args.html_folder
    if not os.path.isdir(html_folder):
        logger.log_message(f"[X] The folder {html_folder} does not exist or is not a directory.", level="warning")
        return
    txt_folder = os.path.join(args.save_dir, "txt")
    os.makedirs(txt_folder, exist_ok=True)

    for file in os.listdir(html_folder):
        if not file.endswith(".html"):
            continue
        
        basename, _ = os.path.splitext(file)
        if not basename.isdigit():
            logger.log_message(f"[!] Skipping file with non-numeric basename: {file}", level="warning")
            continue

        chapter_id = int(basename)
        html_path = os.path.join(html_folder, file)
        logger.log_message(f"[>] Processing chapter {chapter_id} from file {html_path}...")
        chapter_processor.process_chapter(html_path, chapter_id, args.save_image, args.save_dir, args.use_ocr, args.use_freq)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode font and extract chapter content from HTML.")
    parser.add_argument("--html_folder", required=True, help="Folder of HTML files.")
    parser.add_argument("--save_image", action="store_true", help="Save rendered character images for inspection.")
    parser.add_argument("--save_dir", default="output", help="Directory to save output text and optional images.")
    parser.add_argument("--use_ocr", action="store_true", help="Enable OCR for generating font mapping (fallback matching if not enabled).")
    parser.add_argument("--use_freq", action="store_true", help="Use frequency table to rank image vector")

    args = parser.parse_args()

    log = logger.setup_logging("qidian-decoder")
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
