#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML parser utilities for extracting and reconstructing text content
from obfuscated web pages that use randomized fonts and CSS tricks.

Key functionalities:
- Extract SSR pageContext embedded in the HTML
- Parse CSS rules used to manipulate text visibility and order
- Recursively extract <p> paragraph structure from the HTML
- Apply rules to reconstruct real text including handling of:
  - font-size:0 hiding
  - transform: scaleX(-1) (mirrored characters)
  - ::before / ::after content insertion
"""

import json
import logging

import cssutils
from bs4 import BeautifulSoup, Tag

from .logger import log_message

cssutils.log.setLevel(logging.CRITICAL)

def find_ssr_pageContext(html_str: str) -> dict:
    """
    Extracts the SSR-injected pageContext from HTML as a Python dictionary.
    This is typically found in a <script id="vite-plugin-ssr_pageContext"> tag.

    Args:
        html_str (str): Full HTML content as a string.

    Returns:
        dict: Parsed JSON object containing SSR pageContext.
    """
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        script_tag = soup.find('script', {'id': 'vite-plugin-ssr_pageContext'})
        if script_tag:
            json_data = script_tag.string.strip()  # 获取 script 标签内的字符串
            data_dict = json.loads(json_data)  # 解析 JSON 字符串为 Python 字典
            return data_dict
    except Exception as e:
        log_message(f"[X] Error at find_ssr_pageContext: {e}", level="warning")
    return {}

def extract_paragraphs_recursively(html_str: str, chapter_id: int) -> list:
    """
    Extracts paragraph elements under <main id="c-{chapter_id}"> from HTML
    and converts them to a nested data structure for further processing.

    Args:
        html_str (str): Full HTML content.
        chapter_id (int): ID used to locate <main id="c-{chapter_id}">.

    Returns:
        tuple[list, str]: List of parsed <p> paragraph data and end_number string.
    """
    end_number = None

    def parse_element(elem):
        nonlocal end_number
        # 如果是 NavigableString，则直接返回文本（或跳过空白）
        if not isinstance(elem, Tag):
            return None
        result = {
            "tag": elem.name,
            "attrs": dict(elem.attrs),
            "data": []
        }
        if len(elem.name) == 6 and not end_number:
            end_number = elem.name[3:]
        for child in elem.contents:
            if isinstance(child, Tag):
                parsed = parse_element(child)
                if parsed:
                    result["data"].append(parsed)
            else:
                text = child
                if text:
                    result["data"].append(text)
        return result

    soup = BeautifulSoup(html_str, 'html.parser')
    main_id = f'c-{chapter_id}'
    main_tag = soup.find('main', id=main_id)
    if not main_tag:
        return []

    result = []
    for p in main_tag.find_all('p'):
        parsed_p = parse_element(p)
        if parsed_p:
            result.append(parsed_p)

    return result, end_number

def parse_rule(css_str) -> dict:
    """
    Parses relevant CSS rules and transforms them into a structured mapping
    that can later be applied to HTML tags during rendering.

    Handles:
    - font-size:0 (to delete content)
    - transform: scaleX(-1) (mirrored content)
    - ::before / ::after with attr() or hardcoded characters
    - class-specific rules for rendering

    Args:
        css_str (str): Raw CSS string extracted from the HTML.

    Returns:
        dict: {
            "rules": parsed CSS rules,
            "orders": selector rendering order list
        }
    """
    def parse_selector_style(style, selector_str) -> dict:
        data = {}

        font_size = style.getPropertyValue('font-size')
        transform = style.getPropertyValue('transform')
        content = style.getPropertyValue('content')

        if font_size == '0':
            if '::first-letter' in selector_str:
                data['delete-first'] = True
            else:
                data['delete-all'] = True

        if transform == 'scalex(-1)':
            data['transform-x_-1'] = True

        if '::after' in selector_str and content:
            if 'attr(' in content:
                attr_name = content.split('attr(')[1].split(')')[0]
                data['append-end-attr'] = attr_name
            else:
                data['append-end-char'] = content.strip('"').strip("'")

        if '::before' in selector_str and content:
            if 'attr(' in content:
                attr_name = content.split('attr(')[1].split(')')[0]
                data['append-start-attr'] = attr_name
            else:
                data['append-start-char'] = content.strip('"').strip("'")

        return data

    sheet = cssutils.parseString(css_str)
    orders = []
    css_rules = {}

    for rule in sheet:
        if rule.type != rule.STYLE_RULE:
            continue

        selector = rule.selectorText
        style = rule.style

        # 1. 提取 order 属性
        order_val = style.getPropertyValue('order')
        if order_val:
            orders.append((selector, order_val))

        temp = parse_selector_style(style, selector)

        # 2. sy 类处理
        if selector.startswith('.sy-'):
            if 'sy' not in css_rules:
                css_rules['sy'] = {}
            css_rules['sy'][selector[1:]] = temp

        # 3. .p* 类处理
        elif selector.startswith('.p') and ' ' in selector:
            parts = selector.split()
            class_str = parts[0].replace('.', '')
            tag_part = parts[1].split('::')[0]

            if class_str not in css_rules:
                css_rules[class_str] = {}

            if tag_part not in css_rules[class_str]:
                css_rules[class_str][tag_part] = temp
            else:
                css_rules[class_str][tag_part].update(temp)

    # 按照 order 数值升序排序
    orders.sort(key=lambda x: int(x[1]))

    return {
        "rules": css_rules,
        "orders": orders
    }

def render_paragraphs(main_paragraphs, rules, end_number):
    """
    Applies the parsed CSS rules to the paragraph structure and reconstructs the visible text.

    Handles special class styles like .sy-*, text order control, mirrored characters, etc.

    Args:
        main_paragraphs (list): Paragraph structures from extract_paragraphs_recursively.
        rules (dict): Parsed CSS rules from parse_rule().
        end_number (str): HTML tag suffix (e.g. span123 → 123).

    Returns:
        tuple[str, list]: Final reconstructed paragraph string and list of mirrored characters.
    """
    orders = rules.get("orders", [])
    rules = rules.get("rules", {})
    refl_list = []

    def apply_rule(data, rule):
        if rule.get("delete-all", False):
            return ""

        curr_str = ""
        if isinstance(data.get("data"), list) and data["data"]:
            first_data = data["data"][0]
            if isinstance(first_data, str):
                curr_str += first_data

        if rule.get("delete-first", False):
            if len(curr_str) <= 1:
                curr_str = ""
            else:
                curr_str = curr_str[1:]

        curr_str += rule.get("append-end-char", "")

        attr_name = rule.get("append-end-attr", "")
        if attr_name:
            curr_str += data.get("attrs", {}).get(f"{attr_name}{end_number}", "")

        curr_str = rule.get("append-start-char", "") + curr_str

        attr_name = rule.get("append-start-attr", "")
        if attr_name:
            curr_str = data.get("attrs", {}).get(f"{attr_name}{end_number}", "") + curr_str

        if rule.get("transform-x_-1", False):
            refl_list.append(curr_str)
        return curr_str

    paragraphs_str = ''
    for paragraph in main_paragraphs:
        class_list = paragraph.get("attrs", {}).get("class", [])
        p_class_str = next((c for c in class_list if c.startswith('p')), None)
        curr_datas = paragraph.get("data", [])

        ordered_cache = {}
        for data in curr_datas:
            # 文本节点直接加
            if isinstance(data, str):
                paragraphs_str += data
                continue

            if isinstance(data, dict):
                tag = data.get("tag", "")
                attrs = data.get("attrs", {})

                # 跳过 span.review
                if tag == "span" and "class" in attrs and "review" in attrs["class"]:
                    continue

                # sy 类型标签处理
                if tag == "y":
                    tag_class_list = attrs.get("class", [])
                    tag_class = next((c for c in tag_class_list if c.startswith("sy-")), None)

                    if tag_class in rules.get("sy", {}):
                        curr_rule = rules["sy"][tag_class]
                        paragraphs_str += apply_rule(data, curr_rule)
                    continue
                
                if not p_class_str:
                    log_message(f"[!] not find p_class_str: {class_list}", level="warning")
                    continue
                # 普通标签处理，根据 orders 顺序匹配
                for ord_selector, ord_id in orders:
                    tag_name = f"{ord_selector}{end_number}"
                    if data.get("tag") != tag_name:
                        continue
                    curr_rule = rules.get(p_class_str, {}).get(ord_selector)
                    curr_rule = curr_rule if curr_rule else {}
                    ordered_cache[ord_selector] = apply_rule(data, curr_rule)
                    break  # 每个 data 匹配一个 rule 就够了
        # 最后按 orders 顺序拼接
        for ord_selector, ord_id in orders:
            if ord_selector in ordered_cache:
                paragraphs_str += ordered_cache[ord_selector]

        paragraphs_str += '\n\n'  # 每段落之间加空行

    return paragraphs_str, refl_list

def format_chapter(title, paragraphs, authorSay=""):
    """
    Generate formatted chapter text:
    - Begin with the title, followed by two newline characters.
    - Split the paragraphs string into lines, strip each line of leading/trailing whitespace,
      and join non-empty lines with two newline characters.
    - If authorSay is provided, append "\n\n---\n\n" followed by the author's comment.
    
    Parameters:
        title (str): The title of the chapter.
        paragraphs (str): A multi-line string where each line represents a paragraph.
        authorSay (str, optional): The author's comment. Defaults to an empty string.
    
    Returns:
        str: The formatted chapter text.
    """
    formatted_paragraphs = "\n\n".join(line.strip() for line in paragraphs.splitlines() if line.strip())
    result = f"{title}\n\n{formatted_paragraphs}"

    # Append the author's comment if provided.
    if authorSay.strip():
        result += f"\n\n---\n\n{authorSay}"
    return result
