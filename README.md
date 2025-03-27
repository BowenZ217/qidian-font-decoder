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

## 使用方法

```bash
# 克隆项目
git clone https://github.com/BowenZ217/qidian-font-decoder.git
cd qidian-font-decoder

# 执行解析脚本
python decode_font.py https://book.qidian.com/chapter/xxx
```

脚本会: 
1. 下载该章节页面的字体文件
2. 利用 OCR 或字体轮廓比对还原真实字符
3. 输出映射字典和解密后的文本内容

如果出现了无法识别的文字可以进行人工标注, 例如当

```bash
[char] 识别失败:「引」(0x5f15)
[char] 识别失败:「铺」(0x94fa)
```

可以打开 `output/.../unfound` 文件夹, 将图片重命名放入 [known_chars](resources/known_chars/) 文件夹中, 并在 [image_label_map.json](resources/image_label_map.json) 文件内添加对应汉字

成功匹配就会出现:

```bash
[Fallback] 图像匹配成功:「引」→「卜」
[Fallback] 图像匹配成功:「诊」→「一」
```

最后打开 `output/.../output.txt` 检查解密后的文本内容

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
