# -*- coding: utf-8 -*-
# @time  : 2025/11/14 21:23
# @author: 'March 7th'
# @gitee : 'https://gitee.com/StarRail-Evernight'

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.bge import get_embedding


def create_vector_db(txt_path: str, db_path: str) -> None:
    """
    创建向量数据库
    :param txt_path: 文本文件路径
    :param db_path: 保存向量数据库路径
    :return: None
    """
    embedding = get_embedding()
    loader = TextLoader(txt_path, encoding="utf-8")
    documents = loader.load()
    spliter = RecursiveCharacterTextSplitter(
        separators=["，", "。", "\n", "\n\n", "_", "-"],  # 用于文本断句的位置
        chunk_size=2000,  # 分区的大小
        chunk_overlap=500  # 分区与分区的交集大小
    )

    split_out = spliter.split_documents(documents)

    Chroma.from_documents(
        documents=split_out,
        embedding=embedding,
        persist_directory=db_path  # 保存词向量数据库
    )


if __name__ == '__main__':
    create_vector_db(
        txt_path="./data/简易RAG测试.txt",
        db_path="./data/rag_db"
    )
