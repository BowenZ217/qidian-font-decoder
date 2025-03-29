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
import json
import os

from src import font_utils, ocr_utils, html_parser, logger

TEMP_FOLDER = 'temp'
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ----------------------
# Helper function
# ----------------------

def process_chapter(html_path, chapter_id, save_image, save_dir, use_ocr, use_freq):
    if not os.path.exists(html_path):
        logger.log_message(f"[X] File not exist: {html_path}", level="warning")
        return

    txt_path = os.path.join(save_dir, "txt", f"{chapter_id}.txt")
    if os.path.exists(txt_path):
        logger.log_message(f"[!] Chapter {chapter_id} already processed. Skipping.", level="warning")
        return
    output_path = os.path.join(save_dir, str(chapter_id))
    os.makedirs(output_path, exist_ok=True)
    # Load HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        html_str = f.read()

    # Extract embedded fonts and CSS from HTML (SSR React data)
    try:
        ssr_pageContext = html_parser.find_ssr_pageContext(html_str)
        css_str = ssr_pageContext['pageContext']['pageProps']['pageData']['chapterInfo']['css']
        randomFont_str = ssr_pageContext['pageContext']['pageProps']['pageData']['chapterInfo']['randomFont']
        fixedFontWoff2_str = ssr_pageContext['pageContext']['pageProps']['pageData']['chapterInfo']['fixedFontWoff2']
        fixedFontTtf_str = ssr_pageContext['pageContext']['pageProps']['pageData']['chapterInfo']['fixedFontTtf']
        chapterName_str = ssr_pageContext["pageContext"]["pageProps"]["pageData"]["chapterInfo"]["chapterName"]
        authorSay_str = ssr_pageContext["pageContext"]["pageProps"]["pageData"]["chapterInfo"]["authorSay"]
    except Exception as e:
        logger.log_message(f"[X] Fail to get ssr_pageContext: {e}", level="warning")
        return

    # Save / Download Fonts
    randomFont_dict = json.loads(randomFont_str)
    font_list = randomFont_dict['data']
    font_bytes = bytes(font_list)
    randomFont_path = os.path.join(TEMP_FOLDER, 'randomFont.ttf')
    with open(randomFont_path, 'wb') as f:
        f.write(font_bytes)

    fixedFont_path = font_utils.download_font(fixedFontWoff2_str, TEMP_FOLDER)

    # Extract and render paragraphs from HTML with CSS rules
    main_paragraphs, end_number = html_parser.extract_paragraphs_recursively(html_str, chapter_id)
    paragraphs_rules = html_parser.parse_rule(css_str)
    paragraphs_str, refl_list = html_parser.render_paragraphs(main_paragraphs, paragraphs_rules, end_number)

    # Run OCR + fallback mapping
    char_set = set(c for c in paragraphs_str if c not in {' ', '\n', '\u3000'})
    refl_set = set(refl_list)
    char_set = char_set - refl_set
    ocr_utils.init(use_ocr=use_ocr, use_freq=use_freq)
    mapping_result = ocr_utils.generate_font_mapping(
        fixedFont_path,
        randomFont_path,
        char_set,
        refl_set,
        output_path,
        save_image
    )

    # If enabled, save mapping preview images in Markdown format
    if save_image:
        ocr_utils.format_font_mapping_md(mapping_result, output_path)

    # Reconstruct final readable text
    final_paragraphs_str = ocr_utils.apply_font_mapping_to_text(paragraphs_str, mapping_result)
    final_str = html_parser.format_chapter(chapterName_str, final_paragraphs_str, authorSay_str)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(final_str)
    logger.log_message(f"[√] Processed chapter {chapter_id} successfully.")
    return

# ----------------------
# Main process
# ----------------------

def main(args):
    log = logger.setup_logging("qidian-decoder")

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
        process_chapter(html_path, chapter_id, args.save_image, args.save_dir, args.use_ocr, args.use_freq)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode font and extract chapter content from HTML.")
    parser.add_argument("--html_folder", required=True, help="Folder of HTML files.")
    parser.add_argument("--save_image", action="store_true", help="Save rendered character images for inspection.")
    parser.add_argument("--save_dir", default="output", help="Directory to save output text and optional images.")
    parser.add_argument("--use_ocr", action="store_true", help="Enable OCR for generating font mapping (fallback matching if not enabled).")
    parser.add_argument("--use_freq", action="store_true", help="Use frequency table to rank image vector")

    args = parser.parse_args()
    main(args)
