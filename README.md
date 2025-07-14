# 📚 智能PDF阅读器

一个基于LangChain的智能PDF阅读器，支持文档问答、向量搜索和多种语言模型后端。

## ✨ 功能特性

- 📄 **PDF文档处理**: 自动加载和解析PDF文件
- 🔍 **智能问答**: 基于文档内容进行问答
- 🗄️ **向量数据库**: 使用FAISS进行高效的相似性搜索
- 🤖 **多模型支持**: 支持规则系统、本地Transformers、HuggingFace、OpenAI API
- 💻 **命令行工具**: 支持命令行交互
- 🔧 **可配置**: 支持自定义模型和参数

## 🛠️ 技术栈

- **LangChain**: 大语言模型应用框架
- **FAISS**: 高效的向量相似性搜索
- **Sentence Transformers**: 文本嵌入模型
- **PyPDF**: PDF文档处理
- **Transformers**: 本地语言模型
- **HuggingFace**: 模型仓库和推理

## 📦 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd langchain
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装额外依赖（可选）

如果需要使用本地Transformers模型：

```bash
pip install accelerate
```

## 🚀 使用方法

### 命令行版本

1. **基本使用**：
```bash
python cli.py --interactive
```

2. **指定PDF文件夹**：
```bash
python cli.py --pdf-folder pdfs --interactive
```

3. **使用本地嵌入模型**：
```bash
python cli.py --embedding-model D:/Embedding/Embedding --interactive
```

4. **使用HuggingFace模型**：
```bash
python cli.py --embedding-model D:/Embedding/Embedding --llm-model "hf:gpt2" --interactive
```

5. **使用OpenAI API**：
```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key"
python cli.py --llm-model openai --interactive
```

6. **自定义参数**：
```bash
python cli.py \
    --pdf-folder my_pdfs \
    --embedding-model all-MiniLM-L6-v2 \
    --llm-model "hf:gpt2" \
    --chunk-size 500 \
    --chunk-overlap 50 \
    --interactive
```

## 📁 项目结构

```
langchain/
├── cli.py              # 命令行工具
├── pdf_reader.py       # PDF处理核心类
├── llm_interface.py    # 语言模型接口
├── requirements.txt    # 项目依赖
├── README.md          # 项目说明
├── bug.md             # Bug记录和解决方案
├── pdfs/              # PDF文件存储文件夹
└── vectorstore/       # 向量数据库存储文件夹
```

## ⚙️ 配置选项

### 嵌入模型

支持多种Sentence Transformers模型：

- `all-MiniLM-L6-v2` (默认，轻量级)
- `paraphrase-multilingual-MiniLM-L12-v2` (多语言支持)
- `all-mpnet-base-v2` (高质量)
- 本地模型路径 (如 `D:/Embedding/Embedding`)

### 语言模型

支持多种模型类型：

#### 1. 规则系统 (默认)
```bash
python cli.py --llm-model rule_based --interactive
```

#### 2. HuggingFace模型
```bash
# 小模型
python cli.py --llm-model "hf:gpt2" --interactive

# 中文模型 (需要足够内存)
python cli.py --llm-model "hf:THUDM/chatglm2-6b" --interactive

# 对话模型
python cli.py --llm-model "hf:microsoft/DialoGPT-medium" --interactive
```

#### 3. 本地Transformers模型
```bash
python cli.py --llm-model "local:D:/models/your-model" --interactive
```

#### 4. OpenAI API
```bash
export OPENAI_API_KEY="your-api-key"
python cli.py --llm-model openai --interactive
```

### 文档分割参数

- **块大小 (chunk_size)**: 文档分割的块大小，默认1000字符
- **块重叠 (chunk_overlap)**: 相邻块的重叠大小，默认200字符

## 🔧 高级配置

### 自定义嵌入模型

```python
from pdf_reader import PDFReader

# 使用自定义嵌入模型
reader = PDFReader(embedding_model="your-custom-model")
```

### 自定义语言模型

```python
from llm_interface import LLMInterface

# 使用自定义语言模型
llm = LLMInterface(model_name="your-custom-model")
```

## 🐛 常见问题

### 1. 模型下载失败

**问题**: 首次运行时模型下载失败

**解决方案**:
```bash
# 手动下载嵌入模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 或使用本地模型
python cli.py --embedding-model D:/Embedding/Embedding --interactive
```

### 2. 内存不足

**问题**: 处理大文件时内存不足

**解决方案**:
- 减小chunk_size参数
- 使用更小的嵌入模型
- 分批处理文档

### 3. 模型输出无意义

**问题**: 小模型输出无意义内容

**解决方案**:
- 使用更适合的模型
- 简化prompt格式
- 限制context长度

### 4. PDF解析错误

**问题**: 某些PDF文件无法正确解析

**解决方案**:
- 确保PDF文件没有加密
- 尝试使用OCR工具预处理PDF
- 检查PDF文件是否损坏

## 📊 性能优化

### 1. 硬件要求

- **CPU**: 建议4核以上
- **内存**: 建议8GB以上
- **存储**: 根据PDF文件大小而定

### 2. 优化建议

- 使用SSD存储提高I/O性能
- 增加内存以提高处理速度
- 使用GPU加速（如果可用）

### 3. 模型选择建议

- **测试环境**: `hf:gpt2` (小模型，快速测试)
- **中文环境**: `hf:THUDM/chatglm2-6b` (需要16GB+内存)
- **生产环境**: `openai` (API模型，稳定可靠)

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 开发环境设置

1. 克隆项目
2. 创建虚拟环境
3. 安装开发依赖
4. 运行测试

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请提交Issue或联系开发者。 