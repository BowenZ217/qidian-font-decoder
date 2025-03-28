# qidian-font-decoder

> 用于解析起点中文网的自定义字体加密, 恢复真实文本 (逆向字体映射)

该项目通过提取 HTML 中嵌入的字体和 CSS 规则, 生成混淆字体与真实字体之间的字符映射, 最终重构出可读的正文内容

P.S. 最终发现在这项任务中, 当已知大致的字体信息 (例如背景、字体大小等) 时, 使用 `cosine_similarity` 对字符图像向量进行匹配通常会得到更准确的结果

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

> **注意**: 本项目可依赖 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 用于识别字体中的真实字符形状  
> 如果您需要使用 OCR 功能（即启用 `--use_ocr` 参数）, 请按照 PaddleOCR 文档安装相关依赖 (或 cuda 版本):

```bash
python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr
```

当前使用版本:

```bash
paddleocr==2.10.0
paddlepaddle==3.0.0rc1
```

如果不启用 OCR (不使用 `--use_ocr` 参数) , 则无需安装 PaddleOCR 及 paddle 相关包。

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

> TODO: 后续支持直接从网络链接下载章节 HTML (包括 Cookie 处理)

### 4. 执行解析脚本

运行解析脚本进行字体解密与文本重构, 示例命令如下:

```bash
python decode_font.py \
  --html_path chapter.html \
  --chapter_id <chapter_id> \
  --save_image \
  --save_dir output/ \
  --use_ocr \
  --use_freq
```

参数说明: 
- `--html_path`: 指定保存的 HTML 文件路径 (必须) 
- `--chapter_id`: 对应章节的 ID (必须) 
- `--save_image`: 是否保存字符图像 (可选) 
- `--save_dir`: 输出目录, 默认 `output/` (可选) 
- `--use_ocr`: 启用 OCR 模块进行字体映射 (可选), 未添加该参数时, 仅使用 fallback 匹配方法, 无需安装 `PaddleOCR`
- `--use_freq`: 当字符相似度不够高时, 通过字符频率来平衡匹配权重 (可选) 

P.S. `2 extra bytes in post.stringData array` 为 `fontTools` 的提示信息, 大概率可以无视

#### 示例命令

```bash
python decode_font.py --html_path chapter.html --chapter_id 12345678 --save_image --save_dir output/ --use_freq
```

如果有多个 html 则可以将文件命名为 `<chapter_id>.html` 放在同一个文件夹内 (假设为 `data/html` 文件夹) 并运行

```bash
python decode_font_v2.py --html_folder data/html --save_image --save_dir output/ --use_freq
```

### 4.5. 脚本行为说明

脚本会执行以下操作: 

1. 自动下载章节所需字体文件 (包含混淆字体和标准字体) 
2. 还原加密字体对应的真实字符, 使用:
  - 人工标注
  - OCR
  - 字符图像向量匹配
  - 字符频率
3. 生成字符映射表, 并输出解密后的正文内容

### 5. 输出结果

运行成功后将在 `output/{chapter_id}` 目录下生成: 
- `{chapter_id}.txt`: 解码后的正文内容
- `font_mapping.json`: 字符映射字典
- `chars/found/`: OCR 或匹配成功的字符图像 (若启用了 `--save_image`)
- `chars/unfound/`: 未能识别的字符图像 (若启用了 `--save_image`, 便于人工检查和补充)

### 6. 人工辅助匹配失败字符 (可选) 

当部分字符无法通过 OCR 或自动匹配识别时, 终端会显示类似提示: 

```bash
[char] 识别失败:「引」(0x5f15)
[char] 识别失败:「铺」(0x94fa)
```

此时可以手动辅助标注: 

1. 打开失败字符图像所在目录: `output/<chapter_id>/chars/unfound/`
2. 将失败图片移动到 `resources/known_chars/` 目录下, 并重命名 (如 `000001.png` 等)
3. 在 `resources/image_label_map.json` 中添加相应映射, 例如:
  ```json
  {
    "000001.png": "一",
    "000002.png": "二",
    "....png": "..."
  }
  ```

4. 重新运行解析脚本, 即可自动识别这些字符

成功匹配后会显示类似日志: 

```bash
[Fallback] 图像匹配成功:「引」→「卜」
[Fallback] 图像匹配成功:「诊」→「一」
```

### 7. 查看解密结果

最终的解密文本将保存在: 

```
output/<chapter_id>/<chapter_id>.txt
```

请打开该文件, 检查正文内容是否恢复正确

日志文件可打开 `logs/qidian-decoder_<date>.log`

## 附加说明

在本任务中, 由于已知背景和字体大小信息, 使用 `cosine_similarity` 对字符图像向量进行匹配通常能得到更准确的结果

此外, 项目中 OCR 功能 (基于 `PaddleOCR`) 为可选项, 仅在添加 `--use_ocr` 参数时启用

如果不启用 OCR 功能, 则只使用图像 `hash` 或向量匹配进行字符映射, 从而减少对额外依赖的要求

## 例子

项目中已包含三个示例章节文件, 位于 [`examples`](examples/) 文件夹中 (选自起点限免章节)

你可以直接运行以下命令来测试解析效果:

```bash
python decode_font.py --html_path examples/833383226.html --chapter_id 833383226 --save_image --use_freq
```

或者使用其他示例文件:

```bash
python decode_font.py --html_path examples/章节ID.html --chapter_id 章节ID --save_image --use_freq
```

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

该脚本用于生成常用汉字字符列表 (也可包含额外的阿拉伯数字和字母) , 供 OCR 模型测试或数据集构建使用

例如, 运行以下命令: 

```bash
python other_tools/generate_common_chars.py \
  --start 0x4E00 \
  --end 0x9FA5 \
  --extra "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
  --output data/common_chars.txt
```

将生成类似下面的文件内容: 
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

- **--start** 与 **--end** 参数定义了 Unicode 范围 (例如, 0x4E00 到 0x9FA5) 
- **--extra** 参数用于添加额外的字符, 如拉丁字母或数字
- 默认范围 0x4E00–0x9FA5 覆盖了常用汉字

### 生成字符图像

该脚本用于将字体文件中的每个字符渲染成独立的图像, 用于 OCR 测试、训练或比对

示例命令如下: 

```bash
python other_tools/generate_dataset.py \
  --font fonts/SourceHanSansSC-Regular.woff2 \
  --chars data/common_chars.txt \
  --output data/images224 \
  --image_size 224 \
  --font_size 160
```

运行后, 每个字符将被渲染成一张图像, 存放在指定的输出目录下

### 生成字符图像向量

该脚本 `generate_char_vectors.py` 用于从指定字体中生成字符图像向量

脚本流程如下: 

1. **渲染字符图像**: 脚本使用指定的字体, 将每个字符渲染成一个固定尺寸 (如 64×64) 的灰度图像
2. **图像处理**: 将渲染后的图像调整为更小尺寸 (例如 32×32) , 然后将图像扁平化为一维向量, 并进行归一化处理
3. **保存数据**  
   - 将所有字符图像向量保存为 `.npy` 文件。 
   - 同时生成一个 `.txt` 文件, 记录每个向量对应的字符 (每行一个) 

#### 使用示例

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

#### 参数说明

- **--font**: 指定字体文件的路径 (支持 `.ttf`、`.otf`、`.woff2` 等格式) 

- **--start** 与 **--end**: 定义需要生成字符的 Unicode 范围, 例如 `0x4E00` 到 `0x9FA5`

- **--extra**: 附加字符, 用于补充 Unicode 范围外的字符, 如数字、拉丁字母等

- **--output**: 指定输出 `.npy` 文件的路径, 同时会生成一个同名的 `.txt` 文件, 保存字符列表

- **--image_size**: 渲染时使用的画布尺寸, 默认 64 (表示 64×64 的画布) 

- **--resize**: 图像在扁平化前调整为的目标尺寸 (正方形) , 例如 32 表示将图像调整为 32×32

- **--font_size**: 用于渲染字符时的字体大小, 默认 48

#### 输出说明

- **char_vectors.npy**: 保存了所有字符图像的向量, 每个向量为归一化的浮点数数组 (例如, 对于 32×32 的图像, 向量维度为 1024)

- **char_vectors.txt**: 与 `.npy` 文件中向量顺序一致的字符列表, 每个字符占一行

这些向量可用于 字符相似度计算 / 向量检索 / (OCR 模型训练) 等应用

## 声明

本项目仅供学习研究使用, 不得用于任何商业或违法用途。请遵守目标网站的 `robots.txt` 和相关法规。

## Acknowledgements

- [Source Han Sans SC](https://github.com/adobe-fonts/source-han-sans)
- [fontTools](https://github.com/fonttools/fonttools)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [chinese-dictionary](https://github.com/mapull/chinese-dictionary)
