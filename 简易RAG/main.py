from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_ollama import ChatOllama


model_name = "D:/LLM_download/BAAI/bge-small-zh-v1.5"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
bge = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# 1. 加载知识库向量数据库
vector_db = Chroma(
    embedding_function=bge,
    persist_directory="./data/yuanshen_db"
)


# 2. 定义RAG检索工具
@tool
def search_knowledgebase(query: str) -> str:
    """搜索企业知识库获取相关文档片段"""
    docs = vector_db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])


# 3. 创建RAG Agent
agent = create_agent(
    model=ChatOllama(model="qwen3:4b_q4_k_m", base_url="http://localhost:10086", temperature=1.),
    tools=[search_knowledgebase],
    system_prompt="你是企业知识库问答助手，回答问题前必须先调用search_knowledgebase工具获取最新信息，若找不到，则回答”不知道",
)

# 4. 知识库问答
response = agent.invoke({
    "messages": [{"role": "user", "content": "提瓦特大陆是什么？"}]
})

print(response["messages"][-1].content)
