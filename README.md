# qidian-font-decoder

> 用于解析起点中文网的自定义字体加密, 恢复真实文本 (逆向字体映射)

## 项目背景

在看到关于「反爬虫」相关的文章时, 第一次接触到“自定义字体加密”这种手段——网站通过生成特殊字体, 把网页中文字变成“乱码”, 让爬虫抓不到真实内容

出于好奇, 我尝试动手解析这种字体加密机制 (拿起点中文网做尝试), 记录整个过程, 并整理成这个项目

目前已实现: 
- 下载网页对应的 `.woff` / `.ttf` 字体文件
- 解析字体轮廓, 生成映射字典
- 自动将乱码文本还原成真实中文

### 思路

1. 保存网页的两个随机字体文件
2. 处理网页的 css
3. 结合 css + ocr 生成字符对应表
4. 处理 html 正文

## 环境准备

建议使用 Anaconda 创建独立环境, 避免包冲突: 

```bash
conda create -n paddle_ocr_env python=3.9 -y
conda activate paddle_ocr_env
pip install -r requirements.txt
```

> 本项目依赖 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), 用于识别字体中的真实字符形状

请按照 PaddleOCR 文档安装: [doc](https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html)

```bash
python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr
```

当前使用版本:

```bash
paddleocr==2.10.0
paddlepaddle==3.0.0rc1
```

## 使用方法

### 1. 克隆项目

```bash
git clone https://github.com/BowenZ217/qidian-font-decoder.git
cd qidian-font-decoder
```

### 2. 安装依赖

建议使用 Python $\geq$ 3.8, 并创建虚拟环境 (可选): 

```bash
pip install -r requirements.txt
```

### 3. 获取章节 HTML 文件

1. 打开起点小说章节页面, 例如: 
   ```
   https://www.qidian.com/chapter/{book_id}/{chapter_id}
   ```
2. 按 `F12` 打开开发者工具, 切换到 **Elements** 面板
3. 找到顶层标签 `<html translate="no"...>`, 右键点击 $\rightarrow$ **Copy** $\rightarrow$ **Copy element**
4. 将复制内容粘贴到本地文件, 保存为 `chapter.html`

TODO: 脚本保存 `Cookie` 直接使用网路链接

### 4. 执行解析脚本

```bash
python decode_font.py \
  --html_path chapter.html \
  --chapter_id <chapter_id> \
  --save_image \
  --save_dir output/
```

参数说明: 
- `--html_path`: 指定保存的 HTML 文件路径（必须）
- `--chapter_id`: 对应章节的 ID（必须）
- `--save_image`: 是否保存字符图像（可选）
- `--save_dir`: 输出目录, 默认 `output/`（可选）

### 5. 输出结果

运行成功后将在 `output/{chapter_id}` 目录下生成: 
- `output.txt`: 解码后的正文内容
- `font_mapping.json`: 映射字典
- `chars/found/`: OCR 成功的字符图像 (如果启用 `--save_image`)
- `chars/unfound/`: OCR 失败的字符图像 (可用于手动检查)

### 示例命令

```bash
python decode_font.py --html_path chapter.html --chapter_id 12345678 --save_image --save_dir output/
```

### 5. 脚本行为说明

脚本会执行以下操作: 

1. 自动下载章节所需字体文件（包含混淆字体和标准字体）
2. 使用 OCR（或图像轮廓比对）还原加密字体对应的真实字符
3. 生成字符映射表, 并输出解密后的正文内容

### 6. 人工辅助匹配失败字符（可选）

当部分字符无法通过 OCR 自动识别时, 你会看到类似提示: 

```bash
[char] 识别失败:「引」(0x5f15)
[char] 识别失败:「铺」(0x94fa)
```

此时可以手动辅助标注: 

1. 打开失败字符图像路径:   
   `output/<chapter_id>/chars/unfound/`
2. 将失败的图片重命名, 并移动到:   
   `resources/known_chars/` 目录下（如: `000001.png`）
3. 在对应的 `resources/image_label_map.json` 中添加映射: 

```json
{
  "000001.png": "一",
  "000002.png": "二",
  ...
}
```

4. 重新运行脚本, 即可自动识别这些字符

成功匹配会显示: 

```bash
[Fallback] 图像匹配成功:「引」→「卜」
[Fallback] 图像匹配成功:「诊」→「一」
```

### 7. 查看解密结果

最终的解密文本将保存为: 

```
output/<chapter_id>/output.txt
```

你可以打开该文件, 查看完整的正文内容是否恢复正确

## 项目结构

```
qidian-font-decoder/
├── src/
│   ├── __init__.py
│   ├── font_utils.py       # 字体处理逻辑 (下载/解析/映射)
│   ├── ocr_utils.py        # OCR 相关函数 (使用 PaddleOCR)
│   └── html_parser.py      # 起点网页结构分析, 提取字体 URL, CSS 和 乱码文本
├── output/                 # 输出的解密结果、映射表等
├── other_tools/            # 附加工具 (测试、图像生成、数据集构造等)
├── fonts/                  # 字体文件
├── temp/                   # 临时缓存文件
├── decode_font.py          # 主入口脚本
├── requirements.txt        # 依赖包列表
├── README.md               # 项目说明
└── .gitignore
```

## 其他工具 (optional)

除了主功能以外, 本项目还提供一些辅助工具, 比如 **生成中文字符图像数据集**, 用于测试 OCR 模型识别字体字符的准确率

### 生成常用汉字字符集

```bash
python other_tools/generate_common_chars.py \
  --start 0x4E00 \
  --end 0x9FA5 \
  --extra "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
  --output data/common_chars.txt
```

This will create a file like:
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

- The `--start` and `--end` arguments define a Unicode range.
- The `--extra` argument adds optional characters like Latin letters or digits.
- The default range `0x4E00–0x9FA5` covers the most common ~3755 Chinese characters (一级字表).

### 生成训练用字符图像

```bash
python other_tools/generate_dataset.py \
  --font fonts/SourceHanSansSC-Regular.woff2 \
  --chars data/common_chars.txt \
  --output data/images224 \
  --image_size 224 \
  --font_size 160
```

该脚本会将字体文件中的每个字符渲染成单独图像, 用于 OCR 测试或比对

## 声明

本项目仅供学习研究使用, 不得用于任何商业或违法用途。请遵守目标网站的 `robots.txt` 和相关法规。

## Acknowledgements

- [Source Han Sans SC](https://github.com/adobe-fonts/source-han-sans)
- [fontTools](https://github.com/fonttools/fonttools)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
