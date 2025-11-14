from langchain.agents import create_agent
from langchain_chroma import Chroma
from utils.bge import get_embedding
from langchain.tools import tool
from utils.get_llm import get_zhipu

# 1. 加载知识库向量数据库
vector_db = Chroma(
    embedding_function=get_embedding(),
    persist_directory="./data/rag_db"
)


# 2. 定义RAG检索工具
@tool
def search_knowledgebase(query: str) -> str:
    """搜索知识库获取相关文档片段"""
    docs = vector_db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])


# 3. 创建RAG Agent
agent = create_agent(
    model=get_zhipu(),
    tools=[search_knowledgebase],
    system_prompt="你是知识库问答助手，回答问题前必须先调用search_knowledgebase工具获取最新信息，若找不到，则回答”不知道",
)

# 4. 知识库问答
if __name__ == '__main__':
    response = agent.invoke({
        "messages": [{"role": "user", "content": "RAG是什么？"}]
    })

    print(response["messages"][-1].content)
