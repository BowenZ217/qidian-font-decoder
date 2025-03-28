#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Font utils
"""

import os
import requests

from .logger import log_message

def download_font(url: str, folder: str) -> str:
    """
    Download a font file from the specified URL and save it into the given folder.

    Parameters:
        url (str): The URL of the font file to download.
        folder (str): The name of the local folder where the font file should be saved.

    Returns:
        str: The full path to the saved font file if successful, otherwise an empty string.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    os.makedirs(folder, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = os.path.join(folder, filename)
    # Try to download the font file
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        # Write the file in binary mode
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        log_message(f"[✓] 字体已保存到: {filepath}")  # Font saved to: {filepath}
        return filepath
    except Exception as e:
        log_message(f"[X] 下载失败: {e}", level="warning")  # Download failed
    return ""
