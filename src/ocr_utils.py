#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Utils Module

This module provides utility functions for optical character recognition (OCR) and font mapping,
primarily used for decrypting custom font encryption in web pages (e.g., the Qidian website).
It supports the following functionalities:

- Loading and processing known labeled character images using perceptual hashing (phash)
  to build a database of known characters.
- Loading precomputed character image vectors and corresponding labels for vector-based matching,
  leveraging cosine similarity to compare input images.
- Optionally initializing and using PaddleOCR for direct character recognition.
- Providing fallback mechanisms that combine OCR, hash matching, and vector matching to identify
  characters in rendered images.
- Generating a mapping from obfuscated (randomized) fonts to their real characters by rendering
  both reference and obfuscated fonts, then matching the results.
- Formatting and applying the generated font mapping to convert encrypted text into readable text.

The module uses libraries such as Pillow, imagehash, numpy, fontTools, and scikit-learn.
When enabled, it integrates PaddleOCR for advanced OCR capabilities.

This set of utilities is a core component of the qidian-font-decoder project,
designed to help restore real text from dynamically scrambled fonts used as an anti-crawling measure.
"""
import os
import json

import imagehash
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity

from .logger import log_message

# 当前脚本路径（src/ocr_utils.py）
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# 回到项目根目录
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))

# 拼出资源路径
KNOWN_IMAGE_FOLDER = os.path.join(ROOT_DIR, "resources", "known_chars")
KNOWN_MAPPING_JSON = os.path.join(ROOT_DIR, "resources", "image_label_map.json")
VECTOR_NPY_PATH = os.path.join(ROOT_DIR, "resources", "char_vectors.npy")
LABEL_TXT_PATH = os.path.join(ROOT_DIR, "resources", "char_vectors.txt")
CHAR_FREQ_PATH = os.path.join(ROOT_DIR, "resources", "char_freq.json")

IMAGE_FOLDER = 'chars'

# 全局变量
USE_OCR = False
USE_FREQ = False
INIT_CHECK = False
OCR = None
KNOWN_HASH_DB = None
CHAR_VECTOR_DB = None
CHAR_VECTOR_LABELS = None
CHAR_VECTOR_SHAPE = None  # (H, W)
CHAR_FREQ_DB = None

CHAR_CANVAS_SIZE = 64
CHAR_FONT_SIZE = 48

OCR_WEIGHT = 0.05
VECTOR_WEIGHT = 1.0
CANDIDATE_K = 5
CANDIDATE_FACTOR = 5


def init(use_ocr=False, use_freq=False):
    """
    Initial all global values
    """
    global OCR, USE_OCR, USE_FREQ, INIT_CHECK
    if INIT_CHECK:
        return False
    INIT_CHECK = True
    if use_ocr:
        import paddle
        from paddleocr import PaddleOCR
        USE_OCR = True
        # 初始化 OCR, 只用识别模型, 跳过检测
        gpu_available = paddle.device.is_compiled_with_cuda()
        OCR = PaddleOCR(use_angle_cls=False, lang='ch', det=False, use_gpu=gpu_available, show_log=False, rec_model_dir='PP-OCRv4-server')

    state = load_known_images(KNOWN_IMAGE_FOLDER, KNOWN_MAPPING_JSON)
    state = load_known_vector_db() and state
    if use_freq:
        USE_FREQ = True
        state = load_char_freq_db() and state
    return state

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
        log_message(f"[!] Skipping hash DB loading: missing file or folder", level="warning")
        KNOWN_HASH_DB = None
        return False

    try:
        with open(mapping_json_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
    except Exception as e:
        log_message(f"[!] Failed to load label map: {e}", level="warning")
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
            log_message(f"[!] Skipping image {filename}: {e}", level="warning")
            continue

    if not hash_db:
        log_message("[!] No valid images found in hash DB.", level="warning")
        KNOWN_HASH_DB = None
        return False

    KNOWN_HASH_DB = hash_db
    log_message(f"[✓] Loaded {len(hash_db)} image hashes from: {image_folder}")
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

    if not os.path.exists(VECTOR_NPY_PATH) or not os.path.exists(LABEL_TXT_PATH):
        log_message(f"[!] Vector or label file not found. Skipping loading.", level="warning")
        return False

    try:
        CHAR_VECTOR_DB = np.load(VECTOR_NPY_PATH)
        num_chars, dim = CHAR_VECTOR_DB.shape
        side = int(np.sqrt(dim))
        CHAR_VECTOR_SHAPE = (side, side)

        with open(LABEL_TXT_PATH, "r", encoding="utf-8") as f:
            CHAR_VECTOR_LABELS = [line.strip() for line in f]

        log_message(f"[✓] Loaded {num_chars} character vectors from resources (size: {CHAR_VECTOR_SHAPE})")
        return True
    except Exception as e:
        log_message(f"[!] Failed to load known vector DB: {e}", level="warning")
        return False

def load_char_freq_db():
    """
    Loads character frequency data from a JSON file and updates the global CHAR_FREQ_DB.

    Returns:
        bool: True if successfully loaded, False otherwise.
    """
    global CHAR_FREQ_DB

    if not os.path.exists(CHAR_FREQ_PATH):
        log_message(f"[!] Frequency file not found. Skipping loading.", level="warning")
        return False

    try:
        with open(CHAR_FREQ_PATH, "r", encoding="utf-8") as f:
            CHAR_FREQ_DB = json.load(f)
        log_message(f"[✓] Successfully loaded character frequency data from {CHAR_FREQ_PATH}")
        return True
    except json.JSONDecodeError as e:
        log_message(f"[!] JSON decoding error while loading frequency table: {e}", level="warning")
    except Exception as e:
        log_message(f"[!] Unexpected error while loading frequency table DB: {e}", level="warning")
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
        log_message(f"[!] Failed to compute hash for image: {e}", level="warning")
        return None

    best_match = None
    best_distance = threshold + 1
    for h, label in known_hash_db.items():
        dist = abs(img_hash - h)
        if dist < best_distance:
            best_distance = dist
            best_match = label

    return best_match if best_distance <= threshold else None

def match_known_image_v2(img, top_k: int = 1, alpha: float = 0.05):
    """
    Match an input image to known character vectors.

    Args:
        img (PIL.Image): Input image to match.
        top_k (int): Number of top matches to return (default 1).
        alpha (float): Weight factor for the frequency bonus (default 0.05).

    Returns:
        (str, float) or List[Tuple[str, float]]:
            - Top-1 result: returns (char, similarity) or
            - Top-k: list of (char, similarity)
            - If no match found or vectors not loaded: returns "" or [] respectively

    Notes:
        - If vector DB is not loaded, returns default empty result instead of raising.
        - Cosine similarity is used for comparison.
        - The candidate pool is expanded (top_k * 5) and then re-ranked by a composite score.
          The composite score is computed as:
              composite_score = similarity * (6 - frequency)
          where frequency is obtained from CHAR_FREQ_DB (defaulting to 5 if not found),
          so that characters with lower frequency values (i.e. more common) get a boost.
    """
    global CHAR_VECTOR_DB, CHAR_VECTOR_LABELS, CHAR_VECTOR_SHAPE, CHAR_FREQ_DB

    if CHAR_VECTOR_DB is None or CHAR_VECTOR_LABELS is None or CHAR_VECTOR_SHAPE is None:
        return "" if top_k == 1 else []

    try:
        # Preprocess the image.
        img = img.convert("L").resize(CHAR_VECTOR_SHAPE)
        vec = np.array(img).astype(np.float32).flatten() / 255.0

        # Compute cosine similarities.
        sims = cosine_similarity([vec], CHAR_VECTOR_DB)[0]
        top_sim = np.max(sims)
        if not USE_FREQ or top_sim > 0.97:
            top_indices = np.argsort(sims)[-top_k:][::-1]
            results = [(CHAR_VECTOR_LABELS[i], sims[i]) for i in top_indices]
            return results[0] if top_k == 1 else results

        # Determine candidate pool size (e.g., top_k * 5).
        candidate_count = min(len(sims), top_k * CANDIDATE_FACTOR)
        candidate_indices = np.argsort(sims)[-candidate_count:][::-1]

        candidates = []
        max_freq = 5  # The worst-case frequency value.
        for i in candidate_indices:
            char = CHAR_VECTOR_LABELS[i]
            sim = sims[i]
            # Use frequency from CHAR_FREQ_DB; default to 5 (least common) if not found.
            freq = CHAR_FREQ_DB.get(char, 5)
            # Normalize frequency to a bonus in [0, 1] where lower freq gives a higher bonus.
            freq_bonus = (max_freq - freq) / max_freq
            composite_score = sim + alpha * freq_bonus
            candidates.append((char, sim, composite_score))

        # Re-rank candidates by composite score in descending order.
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Prepare final results (return only the character and the original similarity).
        final_results = [(char, sim) for char, sim, _ in candidates[:top_k]]
        return final_results[0] if top_k == 1 else final_results
    except Exception as e:
        log_message(f"[!] Failed to match image: {e}", level="warning")
        return "" if top_k == 1 else []

def recognize_with_fallback(char, img, save_path=None, vector_threshold=0.95, top_k: int = 1):
    """
    Try to recognize a character image using OCR, hash, or vector fallback.

    Args:
        char (str): Original character (for logging)
        img (PIL.Image): Image to recognize
        save_path (str): If given, save the image on success or failure
        vector_threshold (float): Similarity threshold for vector matching
        top_k (int): Number of results

    Returns:
        str  or List[Tuple[str, float]]: Predicted or matched character
            - Top-1 result: returns char
            - Top-k: list of (char, similarity)
            - If no match found or vectors not loaded: returns "" or [] respectively
    """
    # phash fallback
    matched_char = match_known_image(img)
    if matched_char:
        log_message(f"[Fallback] Image hash matched: '{char}' -> '{matched_char}'")
        if save_path:
            img.save(save_path)
        return matched_char if top_k == 1 else [(matched_char, 1.0)]

    # # OCR
    # if USE_OCR:
    #     try:
    #         temp_path = "temp.png"
    #         img.save(temp_path)
    #         ocr_result = OCR.ocr(temp_path, cls=False)
    #         os.remove(temp_path)
    #         if ocr_result and ocr_result[0]:
    #             predicted_char = ocr_result[0][0][1][0]
    #             log_message(f"[OCR] 成功识别:「{char}」->「{predicted_char}」")
    #             if save_path:
    #                 img.save(save_path)
    #             return predicted_char
    #     except Exception as e:
    #         log_message(f"[OCR] 识别出错: {e}", level="warning")

    # vector fallback
    # matched = match_known_image_v2(img)
    # if isinstance(matched, tuple):
    #     matched_char, sim_score = matched
    #     if sim_score >= vector_threshold:
    #         log_message(f"[Fallback] 图像 vector 匹配成功 ({sim_score:.4f}):「{char}」->「{matched_char}」")
    #         if save_path:
    #             img.save(save_path)
    #         return matched_char

    # all failed
    # log_message(f"[char] 识别失败:「{char}」({hex(ord(char))})", level="warning")
    # return None

    candidate_scores = {}

    # OCR candidates (if enabled)
    if USE_OCR:
        ocr_results = None
        try:
            temp_path = "temp.png"
            img.save(temp_path)
            ocr_results = OCR.ocr(temp_path, cls=False)
            os.remove(temp_path)
            if ocr_results and ocr_results[0]:
                # Convert OCR results to a candidate list. Expected format: [[box, (text, confidence)], ...]
                ocr_candidates = []
                for line in ocr_results:
                    for res in line:
                        text, conf = res[1]
                        ocr_candidates.append((text, conf))
                # Sort by confidence in descending order and take top CANDIDATE_K results
                ocr_candidates.sort(key=lambda x: x[1], reverse=True)
                for text, conf in ocr_candidates[:CANDIDATE_K]:
                    candidate_scores[text] = candidate_scores.get(text, 0) + conf * OCR_WEIGHT
                    log_message(f"[OCR] Added candidate: '{text}', OCR confidence: {conf}")
        except Exception as e:
            log_message(f"[OCR] Recognition error ({ocr_results}): {e}", level="warning")

    # Vector candidates
    vector_matches = match_known_image_v2(img, top_k=CANDIDATE_K)
    if isinstance(vector_matches, tuple):
        vector_matches = [vector_matches]
    if vector_matches:
        for v_char, sim_score in vector_matches:
            candidate_scores[v_char] = candidate_scores.get(v_char, 0) + sim_score * VECTOR_WEIGHT
            log_message(f"[Vector] Added candidate: '{v_char}', similarity: {sim_score}")
    else:
        log_message("[Vector] No matching candidate found", level="warning")

    if not candidate_scores:
        log_message(f"[char] Recognition failed: '{char}' ({hex(ord(char))})", level="warning")
        return None if top_k == 1 else []

    # 根据累计得分排序候选项
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    best_candidate, best_score = sorted_candidates[0]
    if best_score < vector_threshold:
        log_message(f"[char] Recognition failed: '{char}' ({hex(ord(char))}), best score: {best_score:.4f} below threshold {vector_threshold}", level="warning")
        return None if top_k == 1 else []
    if save_path:
        img.save(save_path)
    log_message(f"[char] Recognition succeeded ({best_score:.4f}): '{char}' -> '{best_candidate}'")
    return best_candidate if top_k == 1 else sorted_candidates[:top_k]

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
        fixed_render = ImageFont.truetype(fixed_font_path, CHAR_FONT_SIZE)

        random_ttf = TTFont(random_font_path)
        random_cmap = random_ttf.getBestCmap()
        random_chars = {chr(code) for code in random_cmap.keys()}
        random_render = ImageFont.truetype(random_font_path, CHAR_FONT_SIZE)

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
            img = Image.new("L", (CHAR_CANVAS_SIZE, CHAR_CANVAS_SIZE), color=255)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), char, font=render_font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (CHAR_CANVAS_SIZE - w) // 2 - bbox[0]
            y = (CHAR_CANVAS_SIZE - h) // 2 - bbox[1]

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
        log_message(f"[X] An error occurred: {e}", level="warning")
    return {}

def format_font_mapping_md(font_map, output_folder: str):
    """
    Save the font mapping as a Markdown file with corresponding images.

    Parameters:
        font_map (dict): A mapping dictionary { "original character": "recognized character" }.
        output_folder (str): The directory path where the Markdown file will be saved.
    """
    try:
        image_dir = os.path.join(IMAGE_FOLDER, "found")
        out_path = os.path.join(output_folder, "font_mapping.md")
        with open(out_path, "w", encoding="utf-8") as md_file:
            for original_char, recognized_char in font_map.items():
                unicode_hex = hex(ord(original_char))
                image_path = os.path.join(image_dir, f"{unicode_hex}.png").replace("\\", "/")  # for Windows path compatibility

                md_file.write(f"**{original_char}** ({unicode_hex}) -> **{recognized_char}**\n\n")
                md_file.write(f"![{original_char}]({image_path})\n\n")
                md_file.write("---\n\n")

        log_message(f"[✓] Markdown file saved to: {out_path}")
    except Exception as e:
        log_message(f"[X] Error writing Markdown file: {e}", level="warning")
    return

def apply_font_mapping_to_text(text: str, font_map: dict):
    """
    Apply a font mapping to the given text.

    This function iterates over each character in the input text and replaces it with the corresponding
    value from the mapping dictionary if one exists. If a character does not have a mapping, it remains unchanged.

    Args:
        text (str): The input text containing characters to be mapped (e.g., obfuscated font characters).
        font_map (dict): A dictionary mapping characters (keys) to their intended (real) characters (values).

    Returns:
        str: The resulting text after applying the font mapping.
    """
    return ''.join(font_map.get(char, char) for char in text)
