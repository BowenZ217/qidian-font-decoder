#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Utils
"""
import os
import json

import imagehash
import paddle
from fontTools.ttLib import TTFont
from paddleocr import PaddleOCR
from PIL import Image, ImageFont, ImageDraw

# 当前脚本路径（src/ocr_utils.py）
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# 回到项目根目录
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))

# 拼出资源路径
KNOWN_IMAGE_FOLDER = os.path.join(ROOT_DIR, "resources", "known_chars")
KNOWN_MAPPING_JSON = os.path.join(ROOT_DIR, "resources", "image_label_map.json")

# 初始化 OCR, 只用识别模型, 跳过检测
gpu_available  = paddle.device.is_compiled_with_cuda()
OCR = PaddleOCR(use_angle_cls=False, lang='ch', det=False, use_gpu=gpu_available, show_log=False, rec_model_dir='PP-OCRv4-server')
IMAGE_FOLDER = 'chars'

def load_known_images(image_folder, mapping_json_path):
    with open(mapping_json_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    hash_db = {}
    for filename, label in label_map.items():
        img_path = os.path.join(image_folder, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('L')
            img_hash = imagehash.phash(img)
            hash_db[img_hash] = label
    return hash_db

def match_known_image(img, known_hash_db, threshold=5):
    img_hash = imagehash.phash(img)
    best_match = None
    best_distance = threshold + 1
    for h, label in known_hash_db.items():
        dist = abs(img_hash - h)
        if dist < best_distance:
            best_distance = dist
            best_match = label
    if best_distance <= threshold:
        return best_match
    return None

def generate_font_mapping(fixed_font_path, random_font_path, char_set, refl_set, output_path, save_image=False):
    """
    Generate a mapping from encrypted (randomized) font characters to real characters.
    
    # fixed_font    - unicode-range: U+4E00-9FA5;
    # random_font   - unicode-range: U+3400-4DB5;

    This function renders each character using both a reference (fixed) font and
    an obfuscated (random) font. OCR is used to recognize the visual character 
    from rendered images. For characters in the refl_set, a horizontally flipped
    version is also tested to improve matching.

    Parameters:
        fixed_font_path (str): Path to the reference font (e.g., with normal Unicode mappings).
        random_font_path (str): Path to the encrypted/obfuscated font.
        char_set (set): Set of characters to be processed normally.
        refl_set (set): Set of characters to be processed as horizontally flipped.
        output_path (str): Directory to save results and images (if enabled).
        save_image (bool): Whether to save rendered images for inspection.

    Returns:
        dict: A dictionary mapping obfuscated characters (as keys) to real recognized characters.
              Also saves this mapping to a JSON file in the output_path.
    """
    try:
        # Load and render fonts
        fixed_ttf = TTFont(fixed_font_path)
        fixed_cmap = fixed_ttf.getBestCmap()
        fixed_chars = {chr(code) for code in fixed_cmap.keys()}
        fixed_render = ImageFont.truetype(fixed_font_path, 40)

        random_ttf = TTFont(random_font_path)
        random_cmap = random_ttf.getBestCmap()
        random_chars = {chr(code) for code in random_cmap.keys()}
        random_render = ImageFont.truetype(random_font_path, 40)

        known_hash_db = load_known_images(KNOWN_IMAGE_FOLDER, KNOWN_MAPPING_JSON)

        canvas_size = 64

        if save_image:
            # os.makedirs(f"{output_path}/{IMAGE_FOLDER}/found", exist_ok=True)
            # os.makedirs(f"{output_path}/{IMAGE_FOLDER}/unfound", exist_ok=True)
            os.makedirs(os.path.join(output_path, IMAGE_FOLDER, "found"), exist_ok=True)
            os.makedirs(os.path.join(output_path, IMAGE_FOLDER, "unfound"), exist_ok=True)

        mapping_result = {}

        def render_and_ocr(char, is_reflect=False):
            if char in fixed_chars:
                render_font = fixed_render
            elif char in random_chars:
                render_font = random_render
            else:
                return None, False  # Character not in either font

            # Create image and draw character
            img = Image.new("L", (canvas_size, canvas_size), color=255)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), char, font=render_font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (canvas_size - w) // 2 - bbox[0]
            y = (canvas_size - h) // 2 - bbox[1]

            draw.text((x, y), char, fill=0, font=render_font)

            if is_reflect:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            return img, True

        # Process normal characters
        for char in sorted(char_set):
            img, valid = render_and_ocr(char)
            if not valid:
                continue

            temp_path = "temp.png"
            img.save(temp_path)
            ocr_result = OCR.ocr(temp_path, cls=False)
            os.remove(temp_path)

            if ocr_result and ocr_result[0]:
                predicted_char = ocr_result[0][0][1][0]
                mapping_result[char] = predicted_char
                if save_image:
                    # img.save(f"{output_path}/{IMAGE_FOLDER}/found/{hex(ord(char))}.png")
                    img.save(os.path.join(output_path, IMAGE_FOLDER, "found", f"{hex(ord(char))}.png"))
            else:
                # 尝试匹配本地图像库
                matched_char = match_known_image(img, known_hash_db)
                if matched_char:
                    mapping_result[char] = matched_char
                    print(f"[Fallback] 图像匹配成功:「{char}」→「{matched_char}」")
                else:
                    print(f"[char] 识别失败:「{char}」({hex(ord(char))})")
                    if save_image:
                        # img.save(f"{output_path}/{IMAGE_FOLDER}/unfound/{hex(ord(char))}.png")
                        img.save(os.path.join(output_path, IMAGE_FOLDER, "unfound", f"{hex(ord(char))}.png"))

        # Process mirrored characters
        for char in sorted(refl_set):
            img, valid = render_and_ocr(char, is_reflect=True)
            if not valid:
                continue

            temp_path = "temp.png"
            img.save(temp_path)
            ocr_result = OCR.ocr(temp_path, cls=False)
            os.remove(temp_path)

            if ocr_result and ocr_result[0]:
                predicted_char = ocr_result[0][0][1][0]
                mapping_result[char] = predicted_char
                if save_image:
                    # img.save(f"{output_path}/{IMAGE_FOLDER}/found/{hex(ord(char))}_refl.png")
                    img.save(os.path.join(output_path, IMAGE_FOLDER, "found", f"{hex(ord(char))}_refl.png"))
            else:
                # 尝试匹配本地图像库
                matched_char = match_known_image(img, known_hash_db)
                if matched_char:
                    mapping_result[char] = matched_char
                    print(f"[Fallback] 图像匹配成功:「{char}」→「{matched_char}」")
                else:
                    print(f"[refl] 识别失败:「{char}」({hex(ord(char))})")
                    if save_image:
                        # img.save(f"{output_path}/{IMAGE_FOLDER}/unfound/{hex(ord(char))}_refl.png")
                        img.save(os.path.join(output_path, IMAGE_FOLDER, "unfound", f"{hex(ord(char))}_refl.png"))
        
        # Save mapping result to JSON
        # filepath = f"{output_path}/font_mapping.json"
        filepath = os.path.join(output_path, "font_mapping.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping_result, f, ensure_ascii=False, indent=2)
        return mapping_result

    except Exception as e:
        print(f"发生错误: {e}")
    return {}

def format_font_mapping_md(font_map, output_folder: str):
    """
    将字体映射关系保存为 Markdown 文件，结合对应图片展示。

    Parameters:
        font_map (dict): 映射字典 { "原字符": "识别出的字符" }
        output_folder (str): 输出 Markdown 文件路径
    """
    try:
        image_dir = os.path.join(output_folder, IMAGE_FOLDER, "found")
        out_path = os.path.join(output_folder, "font_mapping.md")
        with open(out_path, "w", encoding="utf-8") as md_file:
            for original_char, recognized_char in font_map.items():
                unicode_hex = hex(ord(original_char))
                image_path = os.path.join(image_dir, f"{unicode_hex}.png").replace("\\", "/")  # for Windows path compatibility

                md_file.write(f"**{original_char}** ({unicode_hex}) → **{recognized_char}**\n\n")
                if os.path.exists(image_path):
                    md_file.write(f"![{original_char}]({image_path})\n\n")
                else:
                    md_file.write(f"(未找到图像文件)\n\n")
                md_file.write("---\n\n")

        print(f"✅ Markdown 文件已保存到: {out_path}")

    except Exception as e:
        print(f"❌ 写入 Markdown 文件时出错: {e}")

def fix_paragraphs(paragraphs_str, char_map):
    return ''.join(char_map.get(char, char) for char in paragraphs_str)
