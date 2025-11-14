# 简易RAG系统

一个基于LangChain实现的简易检索增强生成(RAG)系统，用于构建知识库问答应用。

## 项目简介

本项目提供了一个轻量级的RAG系统实现，能够将文本文件转换为向量知识库，并基于知识库内容进行智能问答。系统采用模块化设计，包含向量数据库创建、嵌入模型管理、大语言模型集成和问答服务等核心组件。

## 核心功能

- 文本文件向量化并存储到Chroma向量数据库
- 基于BGE嵌入模型的语义检索
- 集成智谱AI(GLM-4.6)和Ollama本地大模型
- 基于Agent的智能问答系统，自动检索相关知识
- 支持中文文本处理和问答

## 技术栈

- **框架**: LangChain
- **向量数据库**: Chroma
- **嵌入模型**: BGE-small-zh-v1.5 (支持中文的轻量级嵌入模型)
- **大语言模型**: 智谱AI(GLM-4.6) / Ollama本地模型
- **编程语言**: Python 3.11+

## 项目结构

```
简易RAG/
├── main.py                # RAG系统主程序，提供问答功能
├── create_vector_db.py    # 向量数据库创建脚本
├── data/                  # 数据目录
│   ├── 简易RAG测试.txt    # 示例文本文件
│   └── rag_db/            # 向量数据库存储目录
└── utils/                 # 工具模块
    ├── __init__.py
    ├── bge.py            # BGE嵌入模型管理
    └── get_llm.py        # 大语言模型配置和初始化
```

## 功能说明

### 1. 向量数据库创建

`create_vector_db.py`模块负责将文本文件转换为向量数据并存储。主要功能包括：
- 加载文本文件内容
- 使用递归字符分割器将文本分割成适当大小的片段
- 使用BGE嵌入模型生成文本向量
- 将向量和文本存储到Chroma数据库

### 2. 嵌入模型

`utils/bge.py`提供了BGE嵌入模型的初始化和管理功能：
- 默认使用BAAI/bge-small-zh-v1.5中文嵌入模型
- 自动检测并使用GPU加速(如果可用)
- 支持自定义模型路径

### 3. 大语言模型集成

`utils/get_llm.py`支持多种大语言模型的集成：
- 智谱AI(GLM-4.6)模型
- Ollama本地模型(默认qwen3:4b_q4_k_m)
- 可配置温度参数调整生成文本的随机性

### 4. RAG问答系统

`main.py`实现了基于Agent的RAG问答系统：
- 加载预创建的向量数据库
- 定义知识检索工具
- 创建能自动调用工具的Agent
- 提供用户查询接口

## 快速开始

### 环境要求

- Python 3.11+
- 必要的Python包(参见requirements.txt)
- 智谱AI API密钥(如需使用GLM模型)
- 或Ollama本地服务(如需使用本地模型)
- 足够的磁盘空间存储模型和向量数据库

### 安装步骤

1. **克隆项目**
   ```bash
   git clone [项目仓库地址]
   cd 简易RAG
   ```

2. **创建虚拟环境**（推荐）
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   - 对于智谱AI模型，需要配置API密钥
   - 创建`.env`文件并添加以下内容：
     ```
     ZHIPUAI_API_KEY=your_api_key_here
     ```

5. **下载嵌入模型**
   - 默认使用BAAI/bge-small-zh-v1.5模型
   - 可以从Hugging Face下载并放置到默认路径`D:/LLM_download/BAAI/bge-small-zh-v1.5`
   - 或修改`utils/bge.py`中的模型路径

### 使用说明

#### 1. 准备知识库文本

- 在`data`目录下创建或放置文本文件
- 示例：`data/简易RAG测试.txt`

#### 2. 创建向量数据库

```bash
python create_vector_db.py
```

这将自动：
- 读取`data/简易RAG测试.txt`文件
- 将文本分割成合适大小的片段
- 生成向量并存储到`data/rag_db`目录

#### 3. 运行问答系统

```bash
python main.py
```

系统将自动：
- 加载向量数据库
- 初始化大语言模型
- 处理预设问题"RAG是什么？"
- 输出基于知识库的回答

#### 4. 自定义问答

要自定义问答，请修改`main.py`中的查询内容：

```python
response = agent.invoke({
    "messages": [{"role": "user", "content": "你的自定义问题"}]
})
```

## 示例

### 1. 基本示例

#### 创建向量数据库

```python
# 示例：将文本文件转换为向量数据库
from create_vector_db import create_vector_db

# 使用自定义路径
create_vector_db(
    txt_path="./data/我的知识库.txt",
    db_path="./data/custom_rag_db"
)
```

#### 进行知识问答

```python
# 示例：提问关于知识库的问题
from main import agent

# 简单问答
response = agent.invoke({
    "messages": [{"role": "user", "content": "RAG是什么？"}]
})
print(response["messages"][-1].content)

# 复杂问答示例
response = agent.invoke({
    "messages": [{"role": "user", "content": "请详细解释RAG技术的工作原理及其优势"}]
})
print(response["messages"][-1].content)
```

### 2. 高级示例

#### 自定义嵌入模型

```python
from utils.bge import get_embedding

# 使用自定义路径的嵌入模型
custom_embedding = get_embedding(path="./local_models/bge-large-zh")

# 使用自定义嵌入模型创建向量数据库
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/高级文档.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(documents)

# 创建向量数据库
vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=custom_embedding,
    persist_directory="./data/advanced_db"
)
```

#### 切换大语言模型

```python
from utils.get_llm import get_ollama, get_zhipu
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from utils.bge import get_embedding

# 使用Ollama本地模型
ollama_model = get_ollama(model="llama3:8b")

# 创建检索工具
vector_db = Chroma(
    embedding_function=get_embedding(),
    persist_directory="./data/rag_db"
)

@tool
def search_knowledgebase(query: str) -> str:
    """搜索知识库获取相关文档片段"""
    docs = vector_db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

# 创建使用Ollama的Agent
ollama_agent = create_agent(
    model=ollama_model,
    tools=[search_knowledgebase],
    system_prompt="你是知识库问答助手，回答问题前必须先调用search_knowledgebase工具获取最新信息"
)

# 使用Ollama Agent进行问答
response = ollama_agent.invoke({
    "messages": [{"role": "user", "content": "解释向量数据库的原理"}]
})
print(response["messages"][-1].content)

## 注意事项

- 使用智谱AI模型需要配置API密钥
- BGE嵌入模型默认路径为"D:/LLM_download/BAAI/bge-small-zh-v1.5"，可根据实际情况修改
- 文本分割参数(chunk_size和chunk_overlap)可能需要根据文本特点进行调整
- 确保Chroma向量数据库路径有写入权限
- 对于大型知识库，可能需要增加RAM和GPU资源

## 常见问题 (FAQ)

### 1. 安装问题

**Q: 安装依赖时出现错误怎么办？**
A: 确保Python版本在3.8以上，尝试升级pip后重新安装：
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Q: 无法找到嵌入模型怎么办？**
A: 检查模型路径是否正确，或修改`utils/bge.py`中的默认路径：
```python
model_name=path or "your/custom/model/path"
```

### 2. 运行问题

**Q: 运行时报错"ZHIPUAI_API_KEY not set"怎么办？**
A: 需要创建`.env`文件并设置智谱AI的API密钥，或者切换到使用Ollama本地模型。

**Q: Ollama连接失败怎么办？**
A: 确保Ollama服务已启动，检查`base_url`参数是否正确，默认是"http://localhost:10086"。

**Q: 创建向量数据库时报权限错误怎么办？**
A: 确保对目标目录有写入权限，尝试更改数据库保存路径。

### 3. 性能问题

**Q: 嵌入生成速度太慢怎么办？**
A: 确保已正确配置GPU，或减小文本文件大小，调整文本分割参数。

**Q: 检索结果不准确怎么办？**
A: 尝试调整以下参数：
- 增加`k`值获取更多相关文档（默认k=3）
- 调整文本分割的`chunk_size`和`chunk_overlap`
- 使用更大的嵌入模型

### 4. 功能扩展

**Q: 如何支持PDF文件？**
A: 可以使用PyPDFLoader替换TextLoader：
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("your_file.pdf")
```

**Q: 如何修改检索结果数量？**
A: 修改`search_knowledgebase`函数中的`k`参数：
```python
docs = vector_db.similarity_search(query, k=5)  # 获取5个最相关的文档
```

## 扩展与优化

- 支持更多文件格式(PDF、Word等)
- 添加Web界面
- 优化检索算法
- 支持多模态内容
- 实现增量更新知识库功能
- 添加缓存机制提高响应速度
- 实现多轮对话上下文管理

## 许可证

MIT License

## 作者

March 7th
https://gitee.com/StarRail-Evernight