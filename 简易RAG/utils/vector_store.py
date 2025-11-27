# -*- coding: utf-8 -*-
"""
向量存储管理模块，提供增强的向量数据库功能，包括多集合管理、高级检索策略和性能优化
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedVectorStore:
    """
    增强的向量存储管理器，支持多集合、高级检索和性能优化
    """
    
    def __init__(self,
                 embedding_function: HuggingFaceEmbeddings,
                 persist_directory: str,
                 collection_name: str = "default",
                 **kwargs):
        """
        初始化向量存储管理器
        
        :param embedding_function: 嵌入函数
        :param persist_directory: 持久化目录
        :param collection_name: 集合名称
        :param kwargs: 额外配置参数
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 创建持久化目录
        os.makedirs(persist_directory, exist_ok=True)
        
        # 优化的Chroma配置参数
        self.chroma_config = {
            "allow_reset": True,  # 允许重置数据库
            "collection_metadata": {
                "description": f"RAG系统向量集合 - {collection_name}",
                "embedding_model": embedding_function.model_name
            }
        }
        
        # 合并用户提供的额外配置
        self.chroma_config.update(kwargs)
        
        # 初始化向量存储
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            **self.chroma_config
        )
        
        logger.info(f"向量存储初始化完成，集合: {collection_name}，路径: {persist_directory}")
    
    def add_documents(self,
                      documents: List[Document],
                      batch_size: int = 100,
                      persist: bool = True) -> List[str]:
        """
        批量添加文档到向量存储，支持分批处理
        
        :param documents: 文档列表
        :param batch_size: 批处理大小
        :param persist: 是否自动持久化
        :return: 文档ID列表
        """
        if not documents:
            logger.warning("没有要添加的文档")
            return []
        
        logger.info(f"准备添加 {len(documents)} 个文档到集合: {self.collection_name}")
        
        # 文档ID列表
        all_ids = []
        
        # 分批处理大型文档集合
        if len(documents) <= batch_size:
            # 小批量直接处理
            ids = self.vector_store.add_documents(documents)
            all_ids.extend(ids)
        else:
            # 大批量分批处理
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                logger.info(f"处理批次 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                batch_ids = self.vector_store.add_documents(batch_docs)
                all_ids.extend(batch_ids)
        
        # 自动持久化
        if persist:
            self.persist()
            logger.info(f"已添加 {len(all_ids)} 个文档并持久化")
        
        return all_ids
    
    def add_texts(self,
                 texts: List[str],
                 metadatas: Optional[List[Dict[str, Any]]] = None,
                 batch_size: int = 100,
                 persist: bool = True) -> List[str]:
        """
        批量添加文本到向量存储
        
        :param texts: 文本列表
        :param metadatas: 元数据列表
        :param batch_size: 批处理大小
        :param persist: 是否自动持久化
        :return: 文档ID列表
        """
        # 将文本转换为文档对象
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        return self.add_documents(documents, batch_size, persist)
    
    def similarity_search(self,
                         query: str,
                         k: int = 5,
                         filter: Optional[Dict[str, Any]] = None,
                         search_type: str = "similarity",
                         fetch_k: int = 20) -> List[Dict[str, Any]]:
        """
        高级相似度搜索，支持多种检索策略
        
        :param query: 查询文本
        :param k: 返回的文档数量
        :param filter: 元数据过滤条件
        :param search_type: 搜索类型，可选值：similarity, similarity_score_threshold, mmr
        :param fetch_k: MMR搜索时预取的文档数量
        :return: 带分数的文档列表
        """
        logger.info(f"执行检索查询: {query[:50]}...，搜索类型: {search_type}")
        
        results = []
        
        if search_type == "similarity":
            # 基本相似度搜索
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k, filter=filter)
            for doc, score in docs_with_scores:
                results.append({
                    "doc": doc,
                    "score": score
                })
        
        elif search_type == "similarity_score_threshold":
            # 基于阈值的相似度搜索
            # 这里使用一个默认阈值0.5，可以根据实际需要调整
            docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k, filter=filter
            )
            for doc, relevance_score in docs_with_scores:
                # relevance_score范围通常是[0,1]，分数越高越相关
                results.append({
                    "doc": doc,
                    "score": relevance_score
                })
        
        elif search_type == "mmr":
            # 最大边际相关性搜索
            docs = self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, filter=filter
            )
            # MMR不直接返回分数，我们可以手动计算或设置默认值
            for doc in docs:
                results.append({
                    "doc": doc,
                    "score": 0.8  # 占位分数
                })
        
        else:
            raise ValueError(f"不支持的搜索类型: {search_type}")
        
        logger.info(f"检索完成，返回 {len(results)} 个结果")
        return results
    
    def hybrid_search(self,
                     query: str,
                     k: int = 5,
                     filter: Optional[Dict[str, Any]] = None,
                     weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        混合检索策略：结合相似度和MMR搜索
        
        :param query: 查询文本
        :param k: 返回的文档数量
        :param filter: 元数据过滤条件
        :param weight: 相似度搜索的权重 (0.0-1.0)
        :return: 优化后的文档列表
        """
        # 首先获取相似度搜索结果
        similarity_results = self.similarity_search(query, k=k*3, filter=filter, search_type="similarity")
        
        # 然后对这些结果应用MMR进一步优化
        # 提取文档内容
        docs = [item["doc"] for item in similarity_results]
        mmr_docs = self.vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=len(docs), filter=filter
        )
        
        # 构建最终结果，保留原始分数
        mmr_ids = {id(doc.page_content) for doc in mmr_docs}
        results = []
        
        for item in similarity_results:
            if id(item["doc"].page_content) in mmr_ids:
                results.append(item)
                if len(results) >= k:
                    break
        
        return results
    
    def persist(self) -> None:
        """
        显式持久化向量存储
        """
        try:
            self.vector_store.persist()
            logger.info(f"向量存储已持久化到: {self.persist_directory}")
        except Exception as e:
            logger.error(f"持久化失败: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        :return: 统计信息字典
        """
        # 获取Chroma集合
        collection = self.vector_store.get()
        
        # 基本统计
        stats = {
            "collection_name": self.collection_name,
            "document_count": len(collection["ids"]),
            "embedding_dimension": len(collection["embeddings"][0]) if collection["embeddings"] else 0,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_function.model_name
        }
        
        # 提取元数据信息
        if collection["metadatas"]:
            # 统计常见元数据字段
            metadata_fields = set()
            for metadata in collection["metadatas"]:
                metadata_fields.update(metadata.keys())
            stats["metadata_fields"] = list(metadata_fields)
        
        logger.info(f"集合统计: {stats}")
        return stats
    
    def delete_documents(self, ids: List[str], persist: bool = True) -> None:
        """
        删除指定ID的文档
        
        :param ids: 文档ID列表
        :param persist: 是否自动持久化
        """
        if not ids:
            logger.warning("没有要删除的文档ID")
            return
        
        try:
            self.vector_store.delete(ids)
            logger.info(f"已删除 {len(ids)} 个文档")
            
            if persist:
                self.persist()
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            raise
    
    def reset(self) -> None:
        """
        重置向量存储（清空所有内容）
        """
        try:
            self.vector_store.reset()
            logger.info(f"向量存储已重置: {self.collection_name}")
        except Exception as e:
            logger.error(f"重置向量存储失败: {str(e)}")
            raise
    
    @classmethod
    def from_documents(cls,
                      documents: List[Document],
                      embedding: HuggingFaceEmbeddings,
                      persist_directory: str,
                      collection_name: str = "default",
                      batch_size: int = 100,
                      **kwargs) -> 'EnhancedVectorStore':
        """
        从文档创建向量存储实例
        
        :param documents: 文档列表
        :param embedding: 嵌入函数
        :param persist_directory: 持久化目录
        :param collection_name: 集合名称
        :param batch_size: 批处理大小
        :param kwargs: 额外配置参数
        :return: EnhancedVectorStore实例
        """
        # 创建向量存储实例
        vector_store = cls(
            embedding_function=embedding,
            persist_directory=persist_directory,
            collection_name=collection_name,
            **kwargs
        )
        
        # 添加文档
        vector_store.add_documents(documents, batch_size=batch_size)
        
        return vector_store


def create_optimized_retriever(vector_store: EnhancedVectorStore,
                              search_type: str = "hybrid",
                              k: int = 5,
                              **kwargs) -> callable:
    """
    创建优化的检索器函数
    
    :param vector_store: 增强的向量存储实例
    :param search_type: 搜索类型
    :param k: 返回的文档数量
    :param kwargs: 额外参数
    :return: 检索器函数
    """
    def retriever(query: str, **retrieval_kwargs) -> List[Dict[str, Any]]:
        """
        检索器函数
        
        :param query: 查询文本
        :param retrieval_kwargs: 检索参数
        :return: 检索到的文档列表
        """
        # 合并默认参数和传入的参数
        params = {
            "k": k,
            **kwargs,
            **retrieval_kwargs
        }
        
        # 根据搜索类型选择检索方法
        if search_type == "hybrid":
            results = vector_store.hybrid_search(query, **params)
        else:
            results = vector_store.similarity_search(query, search_type=search_type, **params)
        
        return results
    
    # 设置元数据，便于调试和监控
    retriever.__name__ = f"optimized_retriever_{search_type}"
    retriever.search_type = search_type
    retriever.vector_store = vector_store
    
    return retriever


def get_default_vector_store(db_path: str,
                            embedding: HuggingFaceEmbeddings,
                            collection_name: str = "default") -> EnhancedVectorStore:
    """
    获取默认配置的向量存储实例
    
    :param db_path: 数据库路径
    :param embedding: 嵌入函数
    :param collection_name: 集合名称
    :return: 配置好的EnhancedVectorStore实例
    """
    # 默认优化配置
    optimized_config = {
        # 可以根据需要添加更多优化配置
    }
    
    return EnhancedVectorStore(
        embedding_function=embedding,
        persist_directory=db_path,
        collection_name=collection_name,
        **optimized_config
    )


if __name__ == "__main__":
    """
    简单测试代码
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    
    # 测试向量存储功能
    try:
        # 初始化嵌入模型
        embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        
        # 创建测试文档
        test_docs = [
            Document(
                page_content="RAG是检索增强生成的缩写，是一种结合检索和生成模型的技术。",
                metadata={"source": "test_doc.md", "chunk_id": 1}
            ),
            Document(
                page_content="向量数据库用于存储和检索文本的向量表示。",
                metadata={"source": "test_doc.md", "chunk_id": 2}
            ),
            Document(
                page_content="Chroma是一个轻量级的向量数据库，适合本地开发使用。",
                metadata={"source": "test_doc.md", "chunk_id": 3}
            )
        ]
        
        # 创建向量存储
        db_path = "./test_vector_db"
        vector_store = EnhancedVectorStore.from_documents(
            test_docs,
            embedding,
            db_path,
            collection_name="test_collection"
        )
        
        # 获取统计信息
        stats = vector_store.get_collection_stats()
        print(f"统计信息: {stats}")
        
        # 测试检索
        query = "什么是RAG？"
        results = vector_store.similarity_search(query, k=2)
        print(f"\n检索结果 ({query}):")
        for i, result in enumerate(results):
            print(f"\n结果 {i+1} (分数: {result['score']}):")
            print(f"内容: {result['doc'].page_content}")
            print(f"元数据: {result['doc'].metadata}")
        
        # 测试混合检索
        hybrid_results = vector_store.hybrid_search(query, k=2)
        print(f"\n混合检索结果:")
        for i, result in enumerate(hybrid_results):
            print(f"\n结果 {i+1}:")
            print(f"内容: {result['doc'].page_content}")
            
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
