# qidian-font-decoder

[![en](https://img.shields.io/badge/lang-en-red.svg)](./README.md)
[![zh](https://img.shields.io/badge/lang-zh_CN_中文-brightgreen.svg)](./README_cn.md)

> A tool for decoding Qidian Chinese's custom font encryption and recovering the real text (reverse font mapping)

This project extracts fonts embedded in HTML along with CSS rules to generate a mapping between the obfuscated font and the actual font. It ultimately reconstructs the readable text content.

**P.S.** In practice, when general font information (such as background, font size, etc.) is known, matching character image vectors using `cosine_similarity` tends to yield more accurate results than OCR.

## Project Background

After reading several articles about anti-crawling techniques, I first encountered the method of “custom font encryption” — where websites dynamically generate scrambled and encrypted fonts for each request. This process transforms the text into gibberish, thereby preventing crawlers from obtaining the real content.

Out of curiosity, I decided to analyze this font encryption mechanism (using the Qidian Chinese website mentioned in the articles as an example), documented the entire process, and compiled it into this project.

Currently implemented features include:
- Downloading the webpage's corresponding `.woff` / `.ttf` font files
- Parsing font outlines to generate a mapping dictionary
- Automatically restoring obfuscated text back into real Chinese

### Approach

1. Save the two random font files from the webpage
2. Process the webpage's CSS
3. Combine CSS and OCR with image vectors to generate a character mapping table
4. Process the HTML text content

## Environment Setup

It is recommended to use Anaconda to create an isolated environment to avoid package conflicts:

```bash
conda create -n paddle_ocr_env python=3.9 -y
conda activate paddle_ocr_env
pip install -r requirements.txt
```

> **Note**: This project can rely on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to recognize the true character shapes in the fonts.  
> If you need to use the OCR functionality (i.e. enable the `--use_ocr` parameter), please install the required dependencies (or the appropriate CUDA version) according to the PaddleOCR documentation:

```bash
python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr
```

Currently used versions:

```bash
paddleocr==2.10.0
paddlepaddle==3.0.0rc1
```

If OCR is not enabled (i.e. you do not use the `--use_ocr` parameter), there is no need to install PaddleOCR or paddle-related packages.

## Usage

### 1. Clone the Project

```bash
git clone https://github.com/BowenZ217/qidian-font-decoder.git
cd qidian-font-decoder
```

### 2. Install Dependencies

It is recommended to use Python $\geq$ 3.9, and you may create a virtual environment (optional):

```bash
pip install -r requirements.txt
```

### 3. Obtaining the Chapter HTML File

1. Open the Qidian novel chapter page, for example:  
   ```
   https://www.qidian.com/chapter/{book_id}/{chapter_id}
   ```
2. Press `F12` to open the Developer Tools and switch to the **Elements** panel.
3. Locate the top-level `<html translate="no"...>` tag, right-click it -> **Copy** -> **Copy element**.
4. Paste the copied content into a local file and save it as `chapter.html`.

> TODO: In the future, support will be added for directly downloading the chapter HTML from a URL (including cookie handling).

### 4. Running the Decoding Script

Run the decoding script to decrypt the font and reconstruct the text. An example command is as follows:

```bash
python decode_font.py \
  --html_path chapter.html \
  --chapter_id <chapter_id> \
  --save_image \
  --save_dir output/ \
  --use_ocr \
  --use_freq
```

**Parameter Descriptions:**
- `--html_path`: Specifies the path to the saved HTML file (required).
- `--chapter_id`: The ID corresponding to the chapter (required).
- `--save_image`: Whether to save the character images (optional).
- `--save_dir`: The output directory, default is `output/` (optional).
- `--use_ocr`: Enable the OCR module for font mapping (optional). Without this parameter, only the fallback matching method is used, and there is no need to install `PaddleOCR`.
- `--use_freq`: When the character similarity is not high enough, character frequency is used to balance the matching weight (optional).

> **P.S.** The message `2 extra bytes in post.stringData array` is a prompt from `fontTools` and can most likely be ignored.

#### Example Command

```bash
python decode_font.py --html_path chapter.html --chapter_id 12345678 --save_image --save_dir output/ --use_freq
```

If there are multiple HTML files, you can name them as `<chapter_id>.html` and place them in the same folder (for example, in a folder named `data/html`), then run:

```bash
python decode_font_v2.py --html_folder data/html --save_image --save_dir output/ --use_freq
```

### 4.5. Explanation of Script Behavior

The script performs the following actions:

1. Automatically downloads the font files required for the chapter (including both the obfuscated font and the standard font).
2. Restores the real characters corresponding to the encrypted font using:
   - Manual annotation
   - OCR
   - Character image vector matching
   - Character frequency
3. Generates a character mapping table and outputs the decrypted text content.

### 5. Output Results

Upon successful execution, the following files will be generated in the `output/{chapter_id}` directory:
- `{chapter_id}.txt`: The decoded text content.
- `font_mapping.json`: The character mapping dictionary.
- `chars/found/`: Images of characters that were successfully recognized via OCR or matching (if `--save_image` is enabled).
- `chars/unfound/`: Images of characters that could not be recognized (if `--save_image` is enabled, for manual inspection and supplementation).

### 6. Manual Assistance for Unmatched Characters (Optional)

If some characters cannot be recognized through OCR or automatic matching, the terminal will display messages similar to:

```bash
[char] Recognition failed: '引' (0x5f15)
[char] Recognition failed: '铺' (0x94fa)
```

At this point, you can manually assist by annotating the characters:

1. Open the directory containing the images of the failed characters: `output/<chapter_id>/chars/unfound/` (or `found` folder).
2. Move the failed images to the `resources/known_chars/` directory and rename them (for example, `000001.png`, etc.).
3. Add the corresponding mappings in `resources/image_label_map.json`. For example:
   ```json
   {
     "000001.png": "一",
     "000002.png": "二",
     "....png": "..."
   }
   ```
4. Re-run the decoding script, and these characters will be automatically recognized.

Upon successful matching, logs similar to the following will be displayed:

```bash
[Fallback] Image hash matched: '引' -> '卜'
[Fallback] Image hash matched: '诊' -> '一'
```

### 7. Viewing the Decrypted Results

The final decrypted text will be saved at:

```
output/<chapter_id>/<chapter_id>.txt
```

Please open this file and verify that the text content has been restored correctly.

You can also check the log file at `logs/qidian-decoder_<date>.log`.

## Additional Notes

In this task, because the background and font size information is known, using `cosine_similarity` to match character image vectors generally yields more accurate results.

Additionally, the OCR functionality in the project (based on `PaddleOCR`) is optional and is only enabled when the `--use_ocr` parameter is added.

If OCR is not enabled, only image `hash` or vector matching is used for character mapping, thereby reducing the need for additional dependencies.

## Examples

The project includes three example chapter files located in the [`examples`](examples/) folder (selected from Qidian's free chapters).

You can directly run the following command to test the parsing effect:

```bash
python decode_font.py --html_path examples/833383226.html --chapter_id 833383226 --save_image --use_freq
```

Alternatively, use another example file:

```bash
python decode_font.py --html_path examples/<chapter_id>.html --chapter_id <chapter_id> --save_image --use_freq
```

## Project Structure

```
qidian-font-decoder/
├── examples/               # Examples
├── resources/
├── src/
│   ├── __init__.py
│   ├── font_utils.py       # Font processing logic (download/parse/mapping)
│   ├── ocr_utils.py        # OCR-related functions (using PaddleOCR and image vector)
│   └── html_parser.py      # Qidian webpage structure analysis, extracting font URLs, CSS, and obfuscated text
├── output/                 # Output for decrypted results, mapping tables, etc.
├── other_tools/            # Additional tools (testing, image generation, dataset construction, etc.)
├── fonts/                  # Font files
├── temp/                   # Temporary cache files
├── test.ipynb              # Test notebook
├── PaddleOCR.ipynb         # PaddleOCR test notebook
├── decode_font.py          # Main entry script
├── decode_font_v2.py       # Main entry script 2.0
├── requirements.txt        # Dependency list
├── README.md               # Project documentation
└── .gitignore
```

## Other Tools (optional)

In addition to the main functionality, this project also provides some auxiliary tools, such as **generating a dataset of Chinese character images**, which can be used to test the accuracy of OCR models in recognizing font characters.

### Generate a Common Chinese Characters Set

This script is used to generate a list of common Chinese characters (which can also include additional Arabic numerals and letters) for use in OCR model testing or dataset construction.

For example, run the following command:

```bash
python other_tools/generate_common_chars.py \
  --start 0x4E00 \
  --end 0x9FA5 \
  --extra "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
  --output data/common_chars.txt
```

This will generate a file with contents similar to:

```
一
丁
七
万
...
A
B
C
...
9
```

- The **--start** and **--end** parameters define the Unicode range (for example, 0x4E00 to 0x9FA5).
- The **--extra** parameter is used to add extra characters, such as Latin letters or numbers.
- The default range 0x4E00–0x9FA5 covers common Chinese characters.

### Generate Character Images

This script renders each character from a font file into an individual image for OCR testing, training, or comparison.

An example command is as follows:

```bash
python other_tools/generate_dataset.py \
  --font fonts/SourceHanSansSC-Regular.woff2 \
  --chars data/common_chars.txt \
  --output data/images224 \
  --image_size 224 \
  --font_size 160
```

After running, each character will be rendered as an image and stored in the specified output directory.

### Generate Character Image Vectors

The script `generate_char_vectors.py` is used to generate character image vectors from a specified font.

The workflow of the script is as follows:

1. **Render Character Images**: The script uses the specified font to render each character into a fixed-size (e.g., 64×64) grayscale image.
2. **Image Processing**: The rendered image is resized to a smaller dimension (e.g., 32×32), then flattened into a one-dimensional vector and normalized.
3. **Save Data**  
   - All character image vectors are saved as a `.npy` file.  
   - At the same time, a `.txt` file is generated, recording the corresponding character for each vector (one per line).

#### Usage Example

```bash
python other_tools/generate_char_vectors.py \
  --font fonts/SourceHanSansSC-Regular.woff2 \
  --start 0x4E00 \
  --end 0x9FA5 \
  --extra "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
  --output data/char_vectors.npy \
  --image_size 64 \
  --resize 32 \
  --font_size 48
```

#### Parameter Descriptions

- **--font**: Specifies the path to the font file (supports `.ttf`, `.otf`, `.woff2`, etc.).
- **--start** and **--end**: Define the Unicode range for the characters to be generated, for example, `0x4E00` to `0x9FA5`.
- **--extra**: Additional characters to supplement those outside the Unicode range, such as numbers and Latin letters.
- **--output**: Specifies the output path for the `.npy` file; a `.txt` file with the same name will also be generated to save the character list.
- **--image_size**: The canvas size used for rendering, default is 64 (i.e., a 64×64 canvas).
- **--resize**: The target size (square) to which the image is resized before flattening; for example, 32 means the image is resized to 32×32.
- **--font_size**: The font size used when rendering the characters, default is 48.

#### Output Description

- **char_vectors.npy**: Contains the vectors of all character images, where each vector is a normalized array of floating-point numbers (e.g., for a 32×32 image, the vector dimension is 1024).
- **char_vectors.txt**: A list of characters corresponding to the order of vectors in the `.npy` file, one character per line.

These vectors can be used for character similarity calculation, vector retrieval, OCR model training, and other applications.

## Disclaimer

This project is intended for learning and research purposes only and should not be used for any commercial or illegal activities. Please abide by the target website's `robots.txt` and relevant laws and regulations.

## Acknowledgements

- [Source Han Sans SC](https://github.com/adobe-fonts/source-han-sans)
- [fontTools](https://github.com/fonttools/fonttools)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [chinese-dictionary](https://github.com/mapull/chinese-dictionary)
