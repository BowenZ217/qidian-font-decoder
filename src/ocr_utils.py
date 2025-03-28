#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Utils
"""
import os
import json

import imagehash
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity

# 当前脚本路径（src/ocr_utils.py）
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# 回到项目根目录
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))

# 拼出资源路径
KNOWN_IMAGE_FOLDER = os.path.join(ROOT_DIR, "resources", "known_chars")
KNOWN_MAPPING_JSON = os.path.join(ROOT_DIR, "resources", "image_label_map.json")
VECTOR_NPY_PATH = os.path.join(ROOT_DIR, "resources", "char_vectors.npy")
LABEL_TXT_PATH = os.path.join(ROOT_DIR, "resources", "char_vectors.txt")

IMAGE_FOLDER = 'chars'

# 全局变量
USE_OCR = False
OCR = None
KNOWN_HASH_DB = None
CHAR_VECTOR_DB = None
CHAR_VECTOR_LABELS = None
CHAR_VECTOR_SHAPE = None  # (H, W)

def init(use_ocr=False):
    """
    Initial all global values
    """
    global OCR, USE_OCR
    if use_ocr:
        import paddle
        from paddleocr import PaddleOCR
        USE_OCR = True
        # 初始化 OCR, 只用识别模型, 跳过检测
        gpu_available = paddle.device.is_compiled_with_cuda()
        OCR = PaddleOCR(use_angle_cls=False, lang='ch', det=False, use_gpu=gpu_available, show_log=False, rec_model_dir='PP-OCRv4-server')

    hash_db_check = load_known_images(KNOWN_IMAGE_FOLDER, KNOWN_MAPPING_JSON)
    vector_db_check = load_known_vector_db()
    return hash_db_check and vector_db_check

def load_known_images(image_folder, mapping_json_path):
    """
    Loads known labeled character images into a hash map using perceptual hash (phash).

    Args:
        image_folder (str): Folder containing the labeled character images.
        mapping_json_path (str): JSON file that maps image filenames to character labels.

    Returns:
        bool: True if successfully loaded, False otherwise.

    Notes:
        - If either path is missing or loading fails, returns False and does not raise.
        - Result is also stored in global KNOWN_HASH_DB.
    """
    global KNOWN_HASH_DB

    if not os.path.exists(mapping_json_path) or not os.path.isdir(image_folder):
        print(f"[!] Skipping hash DB loading: missing file or folder")
        KNOWN_HASH_DB = None
        return False

    try:
        with open(mapping_json_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
    except Exception as e:
        print(f"[!] Failed to load label map: {e}")
        KNOWN_HASH_DB = None
        return False

    hash_db = {}
    for filename, label in label_map.items():
        img_path = os.path.join(image_folder, filename)
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert('L')
            img_hash = imagehash.phash(img)
            hash_db[img_hash] = label
        except Exception as e:
            print(f"[!] Skipping image {filename}: {e}")
            continue

    if not hash_db:
        print("[!] No valid images found in hash DB.")
        KNOWN_HASH_DB = None
        return False

    KNOWN_HASH_DB = hash_db
    print(f"[✓] Loaded {len(hash_db)} image hashes from: {image_folder}")
    return True

def load_known_vector_db():
    """
    Loads precomputed character vectors and their labels from resources.

    Returns:
        bool: True if successfully loaded, False otherwise.
    
    Notes:
        - Expects 'char_vectors.npy' and 'char_vectors.txt' in 'resources' folder.
        - If missing, the function returns False and does not raise error.
    """
    global CHAR_VECTOR_DB, CHAR_VECTOR_LABELS, CHAR_VECTOR_SHAPE

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
    VECTOR_NPY_PATH = os.path.join(ROOT_DIR, "resources", "char_vectors.npy")
    LABEL_TXT_PATH = os.path.join(ROOT_DIR, "resources", "char_vectors.txt")

    if not os.path.exists(VECTOR_NPY_PATH) or not os.path.exists(LABEL_TXT_PATH):
        print(f"[!] Vector or label file not found. Skipping loading.")
        return False

    try:
        CHAR_VECTOR_DB = np.load(VECTOR_NPY_PATH)
        num_chars, dim = CHAR_VECTOR_DB.shape
        side = int(np.sqrt(dim))
        CHAR_VECTOR_SHAPE = (side, side)

        with open(LABEL_TXT_PATH, "r", encoding="utf-8") as f:
            CHAR_VECTOR_LABELS = [line.strip() for line in f]

        print(f"[✓] Loaded {num_chars} character vectors from resources (size: {CHAR_VECTOR_SHAPE})")
        return True
    except Exception as e:
        print(f"[!] Failed to load known vector DB: {e}")
        return False

def match_known_image(img, known_hash_db=None, threshold=5):
    """
    Match an image to known image hashes.

    Args:
        img (PIL.Image): Input image to match.
        known_hash_db (dict or None): Optional hash db to use, or None to use global.

    Returns:
        str or None: Best matched character label, or None if not found.
    """
    if known_hash_db is None:
        known_hash_db = KNOWN_HASH_DB

    if not known_hash_db:
        return None

    try:
        img_hash = imagehash.phash(img)
    except Exception as e:
        print(f"[!] Failed to compute hash for image: {e}")
        return None

    best_match = None
    best_distance = threshold + 1
    for h, label in known_hash_db.items():
        dist = abs(img_hash - h)
        if dist < best_distance:
            best_distance = dist
            best_match = label

    return best_match if best_distance <= threshold else None

def match_known_image_v2(img, top_k: int = 1):
    """
    Match an input image to known character vectors.

    Args:
        img (PIL.Image): Input image to match.
        top_k (int): Number of top matches to return (default 1).

    Returns:
        (str, float) or List[Tuple[str, float]]:
            - Top-1 result: returns (char, similarity) or
            - Top-k: list of (char, similarity)
            - If no match found or vectors not loaded: returns "" or [] respectively

    Notes:
        - If vector DB is not loaded, returns default empty result instead of raising.
        - Cosine similarity is used for comparison.
    """
    global CHAR_VECTOR_DB, CHAR_VECTOR_LABELS, CHAR_VECTOR_SHAPE

    if CHAR_VECTOR_DB is None or CHAR_VECTOR_LABELS is None or CHAR_VECTOR_SHAPE is None:
        return "" if top_k == 1 else []

    try:
        img = img.convert("L").resize(CHAR_VECTOR_SHAPE)
        vec = np.array(img).astype(np.float32).flatten() / 255.0

        sims = cosine_similarity([vec], CHAR_VECTOR_DB)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        results = [(CHAR_VECTOR_LABELS[i], sims[i]) for i in top_indices]
        return results[0] if top_k == 1 else results
    except Exception as e:
        print(f"[!] Failed to match image: {e}")
        return "" if top_k == 1 else []

def recognize_with_fallback(char, img, save_path=None, vector_threshold=0.95):
    """
    Try to recognize a character image using OCR, hash, or vector fallback.

    Args:
        char (str): Original character (for logging)
        img (PIL.Image): Image to recognize
        save_path (str): If given, save the image on success or failure
        vector_threshold (float): Similarity threshold for vector matching

    Returns:
        str or None: Predicted or matched character, or None if all failed
    """
    # vector fallback
    matched = match_known_image_v2(img)
    if isinstance(matched, tuple):
        matched_char, sim_score = matched
        if sim_score >= vector_threshold:
            print(f"[Fallback] 图像 vector 匹配成功 ({sim_score:.4f}):「{char}」→「{matched_char}」")
            if save_path:
                img.save(save_path)
            return matched_char

    # phash fallback
    matched_char = match_known_image(img)
    if matched_char:
        print(f"[Fallback] 图像 hash 匹配成功:「{char}」→「{matched_char}」")
        if save_path:
            img.save(save_path)
        return matched_char

    # OCR
    if USE_OCR:
        try:
            temp_path = "temp.png"
            img.save(temp_path)
            ocr_result = OCR.ocr(temp_path, cls=False)
            os.remove(temp_path)
            if ocr_result and ocr_result[0]:
                predicted_char = ocr_result[0][0][1][0]
                print(f"[OCR] 成功识别:「{char}」→「{predicted_char}」")
                if save_path:
                    img.save(save_path)
                return predicted_char
        except Exception as e:
            print(f"[OCR] 识别出错: {e}")

    # all failed
    print(f"[char] 识别失败:「{char}」({hex(ord(char))})")
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
        canvas_size = 64   # Canvas size
        font_size = 48     # Font size
        # Load and render fonts
        fixed_ttf = TTFont(fixed_font_path)
        fixed_cmap = fixed_ttf.getBestCmap()
        fixed_chars = {chr(code) for code in fixed_cmap.keys()}
        fixed_render = ImageFont.truetype(fixed_font_path, font_size)

        random_ttf = TTFont(random_font_path)
        random_cmap = random_ttf.getBestCmap()
        random_chars = {chr(code) for code in random_cmap.keys()}
        random_render = ImageFont.truetype(random_font_path, font_size)

        if CHAR_VECTOR_DB is None or CHAR_VECTOR_LABELS is None or CHAR_VECTOR_SHAPE is None or KNOWN_HASH_DB is None:
            init()

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

            found_path = os.path.join(output_path, IMAGE_FOLDER, "found", f"{hex(ord(char))}.png")
            unfound_path = os.path.join(output_path, IMAGE_FOLDER, "unfound", f"{hex(ord(char))}.png")
            matched_char = recognize_with_fallback(
                char,
                img,
                save_path=found_path if save_image else None
            )

            if matched_char:
                mapping_result[char] = matched_char
            else:
                if save_image:
                    img.save(unfound_path)

        # Process mirrored characters
        for char in sorted(refl_set):
            img, valid = render_and_ocr(char, is_reflect=True)
            if not valid:
                continue

            found_path = os.path.join(output_path, IMAGE_FOLDER, "found", f"{hex(ord(char))}_refl.png")
            unfound_path = os.path.join(output_path, IMAGE_FOLDER, "unfound", f"{hex(ord(char))}_refl.png")
            matched_char = recognize_with_fallback(
                char,
                img,
                save_path=found_path if save_image else None
            )

            if matched_char:
                mapping_result[char] = matched_char
            else:
                if save_image:
                    img.save(unfound_path)

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
        image_dir = os.path.join(IMAGE_FOLDER, "found")
        out_path = os.path.join(output_folder, "font_mapping.md")
        with open(out_path, "w", encoding="utf-8") as md_file:
            for original_char, recognized_char in font_map.items():
                unicode_hex = hex(ord(original_char))
                image_path = os.path.join(image_dir, f"{unicode_hex}.png").replace("\\", "/")  # for Windows path compatibility

                md_file.write(f"**{original_char}** ({unicode_hex}) → **{recognized_char}**\n\n")
                md_file.write(f"![{original_char}]({image_path})\n\n")
                md_file.write("---\n\n")

        print(f"✅ Markdown 文件已保存到: {out_path}")

    except Exception as e:
        print(f"❌ 写入 Markdown 文件时出错: {e}")

def fix_paragraphs(paragraphs_str, char_map):
    return ''.join(char_map.get(char, char) for char in paragraphs_str)
