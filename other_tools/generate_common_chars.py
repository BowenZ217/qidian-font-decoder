#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a list of characters (e.g., Chinese + alphanumerics) for training.

Usage:
    python tools/generate_common_chars.py \
        --start 0x4E00 \
        --end 0x9FA5 \
        --extra "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
        --output data/common_chars.txt
"""

import argparse

def generate_char_list(start_idx, end_idx, extra_chars, output_file):
    chars = []

    # Add Chinese characters from Unicode range
    for code in range(start_idx, end_idx + 1):
        chars.append(chr(code))

    # Add any extra characters if specified
    if extra_chars:
        for c in extra_chars:
            if c not in chars:
                chars.append(c)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for c in chars:
            f.write(c + '\n')

    print(f"[✓] Saved {len(chars)} characters to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a character list from a Unicode range and extra characters."
    )
    parser.add_argument(
        "--start", type=lambda x: int(x, 0), default=0x4E00,
        help="Start Unicode code point (e.g., 0x4E00)"
    )
    parser.add_argument(
        "--end", type=lambda x: int(x, 0), default=0x9FA5,
        help="End Unicode code point (e.g., 0x9FA5)"
    )
    parser.add_argument(
        "--extra", type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        help="Extra characters to include"
    )
    parser.add_argument(
        "--output", type=str, default="common_chars.txt",
        help="Output file path"
    )

    args = parser.parse_args()
    generate_char_list(args.start, args.end, args.extra, args.output)
