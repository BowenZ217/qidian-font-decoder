#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate character frequency mapping.

Usage:
    python other_tools/generate_char_freq.py \
        --input resources/char_base.json \
        --output resources/char_freq.json

The input JSON should have the format:
[
  {"index": 52, "char": "反", "strokes": 4, "pinyin": ["fǎn"], "radicals": "又", "frequency": 0, "structure": "R2"}, 
  {"index": 18, "char": "干", "strokes": 3, "pinyin": ["gān", "gàn"], "radicals": "干", "frequency": 0, "structure": "D0", "traditional": "乾幹", "variant": "乹亁榦"},
  {"index": 372, "char": "面", "traditional": "麵", "strokes": 9, "pinyin": ["miàn"], "radicals": "面", "frequency": 0, "structure": "D0", "variant": "靣"},
  {"index": 7467, "char": "砭", "strokes": 9, "pinyin": ["biān"], "radicals": "石", "frequency": 3}
]

Where:
    - "char" represents a unique Chinese character.
    - "frequency" indicates usage frequency:
        0: most common, 1: common, 2: less common, 3: secondary, 4: tertiary, 5: rare.

The output will be a JSON mapping like:
{
  "反": 0,
  "干": 0,
  ...
}
"""

import argparse
import json

def generate_char_freq(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Generate the mapping: {char: frequency}
    char_freq = {}
    for entry in data:
        # Each character is unique as per the input description.
        char_freq[entry['char']] = entry['frequency']
    
    # Save the mapping as a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(char_freq, f, ensure_ascii=False, indent=4)
    print(f"Character frequency mapping saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate character frequency mapping from JSON data.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file with character details")
    parser.add_argument("--output", required=True, help="Path to the output JSON file to save the frequency mapping")
    args = parser.parse_args()
    
    generate_char_freq(args.input, args.output)
