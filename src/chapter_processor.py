#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter Processor Module

This module provides a helper function to process individual chapters from HTML files.
It integrates with several utility modules to perform the following tasks:
  
  - Validate input HTML file paths and manage output directories.
  - Extract embedded fonts, CSS, and chapter metadata from SSR React page context.
  - Download and process font files using custom font utilities.
  - Parse and render chapter paragraphs using defined CSS rules.
  - Generate OCR-based font mapping as a fallback for missing or complex character renderings.
  - Reconstruct the final readable text and output it to a text file.
  - Log relevant processing steps and warnings for debugging purposes.

The main function, `process_chapter`, orchestrates the entire workflow from HTML parsing
to the final text output, ensuring that each chapter is processed only once and that any
missing or problematic data is appropriately handled.
"""

import os
import json

from . import font_utils, ocr_utils, html_parser
from .logger import log_message

TEMP_FOLDER = 'temp'
os.makedirs(TEMP_FOLDER, exist_ok=True)

def process_chapter(html_path, chapter_id, save_image, save_dir, use_ocr, use_freq):
    if not os.path.exists(html_path):
        log_message(f"[X] File not exist: {html_path}", level="warning")
        return

    txt_folder = os.path.join(save_dir, "txt")
    os.makedirs(txt_folder, exist_ok=True)
    txt_path = os.path.join(txt_folder, f"{chapter_id}.txt")
    if os.path.exists(txt_path):
        log_message(f"[!] Chapter {chapter_id} already processed. Skipping.", level="warning")
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
        log_message(f"[X] Fail to get ssr_pageContext: {e}", level="warning")
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
    main_paragraphs = html_parser.extract_paragraphs_recursively(html_str, chapter_id)
    main_paragraphs_path = os.path.join(output_path, f"main_paragraphs_debug.json")
    with open(main_paragraphs_path, 'w', encoding='utf-8') as f:
        json.dump(main_paragraphs, f, ensure_ascii=False, indent=2)
    
    paragraphs_rules = html_parser.parse_rule(css_str)
    paragraphs_rules_path = os.path.join(output_path, f"paragraphs_rules_debug.json")
    with open(paragraphs_rules_path, 'w', encoding='utf-8') as f:
        json.dump(paragraphs_rules, f, ensure_ascii=False, indent=2)

    paragraph_names = html_parser.parse_paragraph_names(paragraphs_rules)
    end_number = html_parser.parse_end_number(main_paragraphs, paragraph_names)
    paragraph_names_path = os.path.join(output_path, f"paragraph_names_debug.txt")
    with open(paragraph_names_path, 'w', encoding='utf-8') as f:
        temp = f"names:\n{paragraph_names}\n\nend_number: {end_number}"
        f.write(temp)

    if not end_number:
        log_message(f"[!] Warning: No end_number found after parsing chapter '{chapter_id}'", level="warning")
        return
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

    debug_path = os.path.join(output_path, f"{chapter_id}_debug.txt")
    with open(debug_path, 'w', encoding='utf-8') as f:
        f.write(paragraphs_str)

    # Reconstruct final readable text
    final_paragraphs_str = ocr_utils.apply_font_mapping_to_text(paragraphs_str, mapping_result)
    final_str = html_parser.format_chapter(chapterName_str, final_paragraphs_str, authorSay_str)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(final_str)
    log_message(f"[√] Processed chapter {chapter_id} successfully.")
    return
