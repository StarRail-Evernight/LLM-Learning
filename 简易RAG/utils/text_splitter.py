# -*- coding: utf-8 -*-
# @time  : 2024/7/1
# @author: 'March 7th'
# @gitee : 'https://gitee.com/StarRail-Evernight'

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_text_splitters.base import TextSplitter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedTextSplitter:
    """
    增强的文本分割器，提供多种分割策略和优化选项
    """
    
    # 针对中文优化的默认分隔符
    CHINESE_SEPARATORS = [
        "\n\n",  # 段落分隔
        "\n",     # 换行
        "。",      # 中文句号
        "！",      # 感叹号
        "？",      # 问号
        "；",      # 分号
        "，",      # 逗号
        " ",       # 空格
        "",        # 最后使用空字符串（强制分割）
    ]
    
    # 针对英文优化的默认分隔符
    ENGLISH_SEPARATORS = [
        "\n\n",  # 段落分隔
        "\n",     # 换行
        ". ",     # 英文句号加空格
        "! ",     # 感叹号加空格
        "? ",     # 问号加空格
        "; ",     # 分号加空格
        ", ",     # 逗号加空格
        " ",       # 空格
        "",        # 最后使用空字符串（强制分割）
    ]
    
    @classmethod
    def create(cls, 
               strategy: str = "recursive", 
               chunk_size: int = 512, 
               chunk_overlap: int = 100,
               language: str = "zh",
               **kwargs) -> TextSplitter:
        """
        创建文本分割器实例
        
        :param strategy: 分割策略
            - "recursive": 递归字符分割（默认）
            - "character": 简单字符分割
            - "token": 基于token的分割
            - "markdown": 基于Markdown结构的分割
        :param chunk_size: 块大小
        :param chunk_overlap: 块重叠大小
        :param language: 语言（"zh"表示中文，"en"表示英文）
        :param kwargs: 其他参数
        :return: TextSplitter实例
        """
        if strategy == "recursive":
            return cls._create_recursive_splitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                language=language,
                **kwargs
            )
        elif strategy == "character":
            return cls._create_character_splitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        elif strategy == "token":
            return cls._create_token_splitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        elif strategy == "markdown":
            return cls._create_markdown_splitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的分割策略: {strategy}")
    
    @classmethod
    def _create_recursive_splitter(cls, 
                                 chunk_size: int = 512, 
                                 chunk_overlap: int = 100,
                                 language: str = "zh",
                                 **kwargs) -> RecursiveCharacterTextSplitter:
        """
        创建递归字符分割器
        """
        # 根据语言选择适当的分隔符
        separators = cls.CHINESE_SEPARATORS if language == "zh" else cls.ENGLISH_SEPARATORS
        
        # 如果用户提供了自定义分隔符，则使用自定义的
        if "separators" in kwargs:
            separators = kwargs["separators"]
        
        # 长度函数，默认使用字符计数
        length_function = kwargs.get("length_function", len)
        
        # 创建分割器
        splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            is_separator_regex=kwargs.get("is_separator_regex", False),
        )
        
        logger.info(f"创建递归字符分割器: 块大小={chunk_size}, 重叠大小={chunk_overlap}, 语言={language}")
        return splitter
    
    @classmethod
    def _create_character_splitter(cls, 
                                 chunk_size: int = 512, 
                                 chunk_overlap: int = 100,
                                 **kwargs) -> CharacterTextSplitter:
        """
        创建简单字符分割器
        """
        separator = kwargs.get("separator", "\n\n")
        length_function = kwargs.get("length_function", len)
        
        splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )
        
        logger.info(f"创建字符分割器: 块大小={chunk_size}, 重叠大小={chunk_overlap}")
        return splitter
    
    @classmethod
    def _create_token_splitter(cls, 
                             chunk_size: int = 512, 
                             chunk_overlap: int = 100,
                             **kwargs) -> SentenceTransformersTokenTextSplitter:
        """
        创建基于token的分割器
        """
        model_name = kwargs.get("model_name", "BAAI/bge-small-zh-v1.5")
        
        splitter = SentenceTransformersTokenTextSplitter(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        logger.info(f"创建Token分割器: 块大小={chunk_size}, 重叠大小={chunk_overlap}, 模型={model_name}")
        return splitter
    
    @classmethod
    def _create_markdown_splitter(cls, 
                                chunk_size: int = 512, 
                                chunk_overlap: int = 100,
                                **kwargs) -> Tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
        """
        创建基于Markdown结构的分割器
        返回两个分割器：一个用于Markdown结构分割，一个用于进一步细分
        """
        # Markdown标题分隔符
        headers_to_split_on = kwargs.get(
            "headers_to_split_on",
            [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        
        # 创建Markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,  # 保留标题文本
        )
        
        # 创建递归字符分割器用于进一步细分
        recursive_splitter = cls._create_recursive_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            language=kwargs.get("language", "zh")
        )
        
        logger.info(f"创建Markdown分割器: 块大小={chunk_size}, 重叠大小={chunk_overlap}")
        return markdown_splitter, recursive_splitter
    
    @staticmethod
    def split_documents(documents: List[Document], 
                       splitter: TextSplitter) -> List[Document]:
        """
        分割文档并添加额外的元数据
        
        :param documents: 文档列表
        :param splitter: 文本分割器
        :return: 分割后的文档列表
        """
        # 分割文档
        split_docs = splitter.split_documents(documents)
        
        # 添加分割相关的元数据
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["chunk_size"] = len(doc.page_content)
            doc.metadata["total_chunks"] = len(split_docs)
            
            # 添加上下文关系
            if i > 0:
                doc.metadata["prev_chunk_id"] = i - 1
            if i < len(split_docs) - 1:
                doc.metadata["next_chunk_id"] = i + 1
        
        logger.info(f"文档分割完成，原始文档数={len(documents)}，分割后文档数={len(split_docs)}")
        return split_docs
    
    @staticmethod
    def split_markdown_documents(documents: List[Document], 
                               markdown_splitter: MarkdownHeaderTextSplitter,
                               content_splitter: TextSplitter) -> List[Document]:
        """
        处理Markdown文档的特殊分割方法
        
        :param documents: 文档列表
        :param markdown_splitter: Markdown结构分割器
        :param content_splitter: 内容分割器
        :return: 分割后的文档列表
        """
        all_splits = []
        
        for doc in documents:
            # 首先使用Markdown分割器按结构分割
            header_splits = markdown_splitter.split_text(doc.page_content)
            
            # 为每个分割后的文档添加原始元数据
            for header_doc in header_splits:
                header_doc.metadata.update(doc.metadata)
                
                # 然后进一步细分内容
                content_splits = content_splitter.split_documents([header_doc])
                all_splits.extend(content_splits)
        
        logger.info(f"Markdown文档分割完成，原始文档数={len(documents)}，分割后文档数={len(all_splits)}")
        return all_splits
    
    @staticmethod
    def optimize_chunk_size(text: str, 
                          target_chunks: int = 10, 
                          min_size: int = 256, 
                          max_size: int = 2048) -> int:
        """
        根据文本长度自动优化块大小
        
        :param text: 示例文本
        :param target_chunks: 目标块数量
        :param min_size: 最小块大小
        :param max_size: 最大块大小
        :return: 优化的块大小
        """
        text_length = len(text)
        
        # 计算理想块大小
        ideal_size = text_length / target_chunks
        
        # 确保块大小在合理范围内
        optimized_size = max(min_size, min(int(ideal_size), max_size))
        
        logger.info(f"优化块大小: 文本长度={text_length}, 目标块数={target_chunks}, 优化后块大小={optimized_size}")
        return optimized_size


def get_optimized_splitter(document_type: str = "general", 
                          chunk_size: Optional[int] = None,
                          chunk_overlap: Optional[int] = None,
                          language: str = "zh") -> TextSplitter:
    """
    获取针对特定文档类型优化的分割器
    
    :param document_type: 文档类型
        - "general": 通用文档
        - "code": 代码文档
        - "report": 报告文档
        - "chat": 对话文档
        - "markdown": Markdown文档
    :param chunk_size: 自定义块大小（可选）
    :param chunk_overlap: 自定义重叠大小（可选）
    :param language: 语言
    :return: 优化的文本分割器
    """
    # 根据文档类型设置默认参数
    type_configs = {
        "general": {
            "strategy": "recursive",
            "chunk_size": chunk_size or 512,
            "chunk_overlap": chunk_overlap or 100,
        },
        "code": {
            "strategy": "recursive",
            "chunk_size": chunk_size or 1024,
            "chunk_overlap": chunk_overlap or 200,
            "separators": [
                "\n\n", "\n", "\nclass ", "\ndef ", "\n#", "\n", " ", ""
            ]
        },
        "report": {
            "strategy": "recursive",
            "chunk_size": chunk_size or 768,
            "chunk_overlap": chunk_overlap or 150,
        },
        "chat": {
            "strategy": "character",
            "chunk_size": chunk_size or 256,
            "chunk_overlap": chunk_overlap or 50,
            "separator": "\n\n",
        },
        "markdown": {
            "strategy": "recursive",
            "chunk_size": chunk_size or 640,
            "chunk_overlap": chunk_overlap or 120,
        }
    }
    
    # 获取配置
    config = type_configs.get(document_type, type_configs["general"])
    
    # 如果用户提供了自定义参数，覆盖默认值
    if chunk_size is not None:
        config["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        config["chunk_overlap"] = chunk_overlap
    
    # 创建分割器
    splitter = EnhancedTextSplitter.create(
        **config,
        language=language
    )
    
    logger.info(f"为文档类型 '{document_type}' 创建优化分割器: {config}")
    return splitter
