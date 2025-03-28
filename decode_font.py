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
    python decode_font.py --html_path chapter.html --chapter_id 1 --save_image --save_dir output/ --use_ocr
"""

import argparse
import json
import os

from src import font_utils, ocr_utils, html_parser

TEMP_FOLDER = 'temp'
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ----------------------
# Main process
# ----------------------

def main(args):
    if not os.path.exists(args.html_path):
        print(f"File not exist: {args.html_path}")
        return
    
    output_path = os.path.join(args.save_dir, str(args.chapter_id))
    os.makedirs(output_path, exist_ok=True)
    # Load HTML content
    with open(args.html_path, 'r', encoding='utf-8') as f:
        html_str = f.read()

    # Extract embedded fonts and CSS from HTML (SSR React data)
    ssr_pageContext = html_parser.find_ssr_pageContext(html_str)
    css_str = ssr_pageContext['pageContext']['pageProps']['pageData']['chapterInfo']['css']
    randomFont_str = ssr_pageContext['pageContext']['pageProps']['pageData']['chapterInfo']['randomFont']
    fixedFontWoff2_str = ssr_pageContext['pageContext']['pageProps']['pageData']['chapterInfo']['fixedFontWoff2']
    fixedFontTtf_str = ssr_pageContext['pageContext']['pageProps']['pageData']['chapterInfo']['fixedFontTtf']

    # Save / Download Fonts
    randomFont_dict = json.loads(randomFont_str)
    font_list = randomFont_dict['data']
    font_bytes = bytes(font_list)
    randomFont_path = os.path.join(TEMP_FOLDER, 'randomFont.ttf')
    with open(randomFont_path, 'wb') as f:
        f.write(font_bytes)

    fixedFont_path = font_utils.download_font(fixedFontWoff2_str, TEMP_FOLDER)

    # Extract and render paragraphs from HTML with CSS rules
    main_paragraphs, end_number = html_parser.extract_paragraphs_recursively(html_str, args.chapter_id)
    paragraphs_rules = html_parser.parse_rule(css_str)
    paragraphs_str, refl_list = html_parser.render_paragraphs(main_paragraphs, paragraphs_rules, end_number)

    # Run OCR + fallback mapping
    char_set = set(c for c in paragraphs_str if c not in {' ', '\n', '\u3000'})
    refl_set = set(refl_list)
    char_set = char_set - refl_set
    ocr_utils.init(use_ocr=args.use_ocr)
    mapping_result = ocr_utils.generate_font_mapping(
        fixedFont_path,
        randomFont_path,
        char_set,
        refl_set,
        output_path,
        args.save_image
    )

    # If enabled, save mapping preview images in Markdown format
    if args.save_image:
        ocr_utils.format_font_mapping_md(mapping_result, output_path)

    # Reconstruct final readable text
    final_str = ocr_utils.apply_font_mapping_to_text(paragraphs_str, mapping_result)
    final_path = os.path.join(output_path, "output.txt")
    with open(final_path, 'w', encoding='utf-8') as f:
        f.write(final_str)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode obfuscated font HTML and reconstruct readable text.")
    parser.add_argument("--html_path", required=True, help="Path to the input HTML file.")
    parser.add_argument("--chapter_id", type=int, required=True, help="Chapter ID used for recursive parsing.")
    parser.add_argument("--save_image", action="store_true", help="Save rendered character images for inspection.")
    parser.add_argument("--save_dir", default="output", help="Directory to save output text and optional images.")
    parser.add_argument("--use_ocr", action="store_true", help="Enable OCR for generating font mapping (fallback matching if not enabled).")

    args = parser.parse_args()
    main(args)
