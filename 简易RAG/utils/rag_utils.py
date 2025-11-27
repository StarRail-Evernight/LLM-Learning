# -*- coding: utf-8 -*-
"""
RAG系统工具函数模块，提供上下文管理、查询优化和响应生成等功能
"""
import re
import logging
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from .vector_store import create_optimized_retriever

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def query_rewriter(original_query: str, chat_history: List[Dict[str, str]] = None, expand_query: bool = True) -> str:
    """
    增强的查询重写和优化，提升检索准确性
    
    :param original_query: 原始查询
    :param chat_history: 对话历史，用于上下文理解
    :param expand_query: 是否执行查询扩展
    :return: 优化后的查询
    """
    # 1. 基础清理
    optimized_query = original_query.strip()
    
    # 2. 移除多余空格和标点
    optimized_query = re.sub(r'\s+', ' ', optimized_query)
    optimized_query = re.sub(r'([。，！？；：])\1+', '\1', optimized_query)
    
    # 3. 处理常见缩写和术语规范化
    abbreviation_map = {
        "RAG": "检索增强生成",
        "LLM": "大语言模型",
        "AI": "人工智能",
        "API": "应用程序编程接口",
        "NLP": "自然语言处理",
        "NER": "命名实体识别"
    }
    
    normalized_query = optimized_query
    for abbr, full in abbreviation_map.items():
        # 使用单词边界确保精确匹配
        pattern = r'\b' + re.escape(abbr) + r'\b'
        normalized_query = re.sub(pattern, f"{full} ({abbr})", normalized_query, flags=re.IGNORECASE)
    
    # 4. 查询扩展 - 添加领域相关词汇
    if expand_query:
        domain_keywords = {
            "RAG": ["检索增强生成", "知识库问答", "向量检索", "生成式问答"],
            "向量数据库": ["向量存储", "embedding", "向量检索", "语义搜索"],
            "大模型": ["LLM", "大语言模型", "生成式AI", "基础模型"],
            "检索": ["查询", "搜索", "查找", "匹配"],
            "优化": ["改进", "提升", "增强", "完善"]
        }
        
        # 提取查询中的关键词
        query_keywords = set(re.findall(r'[\w\u4e00-\u9fa5]+', normalized_query.lower()))
        
        # 查找匹配的领域关键词并扩展
        expanded_terms = set()
        for keyword, related_terms in domain_keywords.items():
            if keyword.lower() in query_keywords:
                expanded_terms.update(related_terms)
        
        # 添加扩展术语（避免重复）
        if expanded_terms:
            # 构建扩展查询，保持原始查询的核心部分
            # 这里使用OR连接扩展术语，让检索系统有更多匹配机会
            expanded_clause = " OR ".join(expanded_terms)
            optimized_query = f"{normalized_query} ({expanded_clause})"
        else:
            optimized_query = normalized_query
    else:
        optimized_query = normalized_query
    
    # 5. 对话历史理解和代词解析
    if chat_history and len(chat_history) > 0:
        # 提取最近的对话内容
        recent_history = " ".join([msg.get("content", "") for msg in chat_history[-3:]])
        
        # 简单的代词解析（实际应用中可以使用更复杂的NLP技术）
        pronouns = {"它": [], "这个": [], "那个": [], "他们": [], "它们": []}
        
        # 从历史对话中提取可能的指代实体
        # 这里使用简单的规则，提取最近提到的名词短语
        potential_references = re.findall(r'[\u4e00-\u9fa5]{2,}', recent_history)
        
        # 如果查询中包含代词，尝试替换为最近的相关实体
        for pronoun in pronouns:
            if pronoun in optimized_query and potential_references:
                # 简单策略：使用最后提到的可能实体
                reference = potential_references[-1]
                optimized_query = optimized_query.replace(pronoun, reference)
                logger.info(f"代词解析: '{pronoun}' -> '{reference}'")
    
    logger.info(f"查询优化: '{original_query}' -> '{optimized_query}'")
    return optimized_query


def context_organizer(retrieved_docs: List[Dict[str, Any]], max_context_length: int = 4000, prioritize_relevance: bool = True) -> str:
    """
    增强的上下文组织器，支持智能排序、去重和上下文优化
    
    :param retrieved_docs: 检索到的文档列表，每个文档包含doc和score
    :param max_context_length: 最大上下文长度
    :param prioritize_relevance: 是否优先考虑相关性（如果为False，则考虑文档顺序）
    :return: 格式化的高质量上下文文本
    """
    if not retrieved_docs:
        logger.warning("没有检索到文档，返回空上下文")
        return ""
    
    # 1. 预处理文档，确保格式一致
    processed_docs = []
    for item in retrieved_docs:
        if isinstance(item, Document):
            processed_docs.append({'doc': item, 'score': 0.0})
        elif isinstance(item, dict) and 'doc' in item:
            # 确保有score字段
            if 'score' not in item:
                item['score'] = 0.0
            processed_docs.append(item)
        else:
            logger.warning(f"无效的文档格式: {item}")
    
    # 2. 智能排序策略
    if prioritize_relevance and 'score' in processed_docs[0]:
        # 优先按相关性排序
        sorted_docs = sorted(processed_docs, key=lambda x: x['score'], reverse=True)
        logger.info("使用相关性排序策略")
    else:
        # 检查是否有位置信息，如果有，尝试按文档顺序排列
        has_position_info = any('chunk_id' in doc['doc'].metadata for doc in processed_docs)
        
        if has_position_info:
            # 按文档来源和chunk_id排序，保持文档的原始顺序
            sorted_docs = sorted(
                processed_docs,
                key=lambda x: (
                    x['doc'].metadata.get('source', ''),
                    int(x['doc'].metadata.get('chunk_id', 0))
                )
            )
            logger.info("使用文档顺序排序策略")
        else:
            # 默认按相关性排序
            sorted_docs = sorted(processed_docs, key=lambda x: x['score'], reverse=True)
    
    # 3. 高级去重和冗余检测
    unique_contents = set()
    organized_context = []
    current_length = 0
    
    # 上下文引入语
    context_header = "【上下文信息】\n"
    current_length += len(context_header)
    organized_context.append(context_header)
    
    for item in sorted_docs:
        doc = item['doc']
        content = doc.page_content.strip()
        
        # 检查完整重复
        if content in unique_contents:
            logger.debug(f"跳过重复文档: {doc.metadata.get('source', '未知')} - {doc.metadata.get('chunk_id', '未知')}")
            continue
        
        # 检查部分重复（如果内容大部分已存在，跳过）
        is_partially_duplicate = False
        for existing in unique_contents:
            # 如果当前内容有80%以上包含在已存在的内容中，视为部分重复
            common_words = set(re.findall(r'[\w\u4e00-\u9fa5]+', content))
            existing_words = set(re.findall(r'[\w\u4e00-\u9fa5]+', existing))
            
            if existing_words and len(common_words.intersection(existing_words)) / len(common_words) > 0.8:
                is_partially_duplicate = True
                break
        
        if is_partially_duplicate:
            logger.debug(f"跳过部分重复文档: {doc.metadata.get('source', '未知')}")
            continue
        
        # 构建丰富的元数据信息
        source_info = f"【来源: {doc.metadata.get('source', '未知')}"
        
        # 添加更多有用的元数据
        if 'page' in doc.metadata:
            source_info += f"，页码: {doc.metadata['page']}"
        if 'chunk_id' in doc.metadata:
            source_info += f"，片段ID: {doc.metadata['chunk_id']}"
        if 'score' in item:
            # 格式化分数显示
            relevance_score = item['score']
            if relevance_score > 0:  # Chroma的分数可能为负数
                source_info += f"，相关度: {relevance_score:.2f}"
        if 'text_relevance' in item:
            source_info += f"，文本相关度: {item['text_relevance']:.2f}"
        
        source_info += "】"
        
        # 计算当前文档块的长度
        doc_block = f"\n{source_info}\n{content}\n"
        doc_length = len(doc_block)
        
        # 智能截断策略
        if current_length + doc_length > max_context_length:
            # 计算剩余空间
            remaining_space = max_context_length - current_length
            
            if remaining_space > 100:  # 只有剩余空间足够时才尝试截断
                # 尝试在句子边界截断
                truncated_content = content[:remaining_space - len(source_info) - 10]
                last_punctuation = max(
                    truncated_content.rfind('。'),
                    truncated_content.rfind('！'),
                    truncated_content.rfind('？'),
                    truncated_content.rfind('.'),
                    truncated_content.rfind('!'),
                    truncated_content.rfind('?')
                )
                
                if last_punctuation > 0:
                    truncated_content = truncated_content[:last_punctuation + 1] + "..."
                else:
                    truncated_content = truncated_content + "..."
                
                # 重新构建截断后的文档块
                truncated_block = f"\n{source_info}\n{truncated_content}\n"
                organized_context.append(truncated_block)
                current_length += len(truncated_block)
                logger.info(f"截断文档以适应上下文长度限制")
                break
            else:
                logger.debug(f"上下文空间不足，跳过文档")
                continue
        
        # 添加到上下文中
        organized_context.append(doc_block)
        unique_contents.add(content)
        current_length += doc_length
    
    # 添加上下文结束标记
    context_footer = "\n【上下文结束】"
    organized_context.append(context_footer)
    current_length += len(context_footer)
    
    logger.info(f"上下文组织完成，长度: {current_length}，文档数: {len(organized_context) - 2}（不包括头部和尾部）")
    
    # 格式化最终上下文
    return "\n".join(organized_context)


def create_enhanced_prompt() -> ChatPromptTemplate:
    """
    创建增强版提示词模板，提升响应质量
    
    :return: 优化的ChatPromptTemplate
    """
    # 系统提示词，提供详细的指令
    system_prompt = """
    你是一个专业的知识库问答助手，你的任务是基于提供的上下文信息，准确回答用户的问题。
    
    工作指南：
    1. 严格基于提供的上下文信息回答问题，不要添加上下文之外的信息或假设
    2. 回答应当清晰、准确、全面，同时保持简洁
    3. 如果上下文信息中包含多个来源，请综合所有相关信息给出答案
    4. 在回答中引用相关来源（如果有来源信息）
    5. 如果上下文信息不足以回答用户问题，或问题与上下文无关，请直接回答"根据提供的信息，我无法回答这个问题"
    6. 对于复杂问题，尝试分步骤或分点回答，提高可读性
    7. 保持专业、友好的语气
    """
    
    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "\n\n上下文信息:\n{context}")
    ])
    
    return prompt


def response_postprocessor(response: str, query: str) -> str:
    """
    响应后处理，提升输出质量
    
    :param response: 原始响应
    :param query: 用户查询
    :return: 处理后的响应
    """
    # 1. 清理响应
    processed_response = response.strip()
    
    # 2. 修复常见问题
    # 移除开头的冗余信息
    processed_response = re.sub(r'^[\s\n]*根据上下文信息，?', '', processed_response)
    processed_response = re.sub(r'^[\s\n]*基于提供的信息，?', '', processed_response)
    
    # 3. 添加引用标记（如果响应中提到了来源）
    # 这里可以实现更复杂的引用提取和格式化逻辑
    
    # 4. 确保回答直接针对问题
    # 检查是否完全回答了问题
    # 这里可以添加更复杂的逻辑来评估回答的相关性和完整性
    
    return processed_response


def create_advanced_retrieval_chain(llm, retriever, search_type: str = "hybrid", rerank: bool = True):
    """
    创建高级检索链，支持多种搜索策略和重排序
    
    :param llm: 语言模型
    :param retriever: 检索器或向量存储实例
    :param search_type: 搜索类型，可选值：similarity, mmr, hybrid
    :param rerank: 是否执行文档重排序
    :return: 增强的检索链
    """
    # 如果传入的是向量存储而非检索器，则创建优化的检索器
    if hasattr(retriever, 'similarity_search'):  # 判断是否是向量存储实例
        retriever = create_optimized_retriever(
            vector_store=retriever,
            search_type=search_type,
            k=5
        )
    
    # 自定义检索函数，集成重排序功能
    def enhanced_retrieval(query, chat_history=None):
        # 使用原始检索器获取文档
        documents = retriever.get_relevant_documents(query)
        
        # 如果启用了重排序，对文档进行重排序
        if rerank and documents:
            # 格式化文档以适应重排序函数
            formatted_docs = []
            for i, doc in enumerate(documents):
                # 检查是否已经有分数
                if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                    score = doc.metadata['score']
                else:
                    score = 1.0 / (i + 1)  # 简单的位置分数
                
                formatted_docs.append({
                    'doc': doc,
                    'score': score
                })
            
            # 执行重排序
            reranked_results = rerank_documents(query, formatted_docs)
            
            # 提取重排序后的文档
            final_docs = [result['doc'] for result in reranked_results]
            logger.info(f"执行文档重排序，从{len(documents)}个文档中选择{len(final_docs)}个最佳文档")
        else:
            final_docs = documents
        
        return final_docs
    
    # 创建历史感知检索器
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=create_enhanced_prompt(),
    )
    
    # 替换检索器的检索方法
    original_get_relevant_documents = history_aware_retriever.get_relevant_documents
    
    def wrapped_retriever(query, **kwargs):
        logger.info(f"执行增强检索，查询: '{query}'")
        if rerank:
            # 获取聊天历史（如果有）
            chat_history = kwargs.get('chat_history', None)
            return enhanced_retrieval(query, chat_history)
        else:
            return original_get_relevant_documents(query, **kwargs)
    
    history_aware_retriever.get_relevant_documents = wrapped_retriever
    
    # 创建文档链
    document_chain = create_stuff_documents_chain(llm, create_enhanced_prompt())
    
    # 创建检索链
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    # 添加搜索类型元数据和配置
    retrieval_chain.search_type = search_type
    retrieval_chain.retriever = retriever
    retrieval_chain.rerank_enabled = rerank
    
    logger.info(f"高级检索链创建完成，搜索类型: {search_type}，重排序: {'开启' if rerank else '关闭'}")
    
    return retrieval_chain


def calculate_relevance_score(query: str, document: str, method: str = "hybrid") -> float:
    """
    增强的相关性评分算法，支持多种评分方法
    
    :param query: 查询文本
    :param document: 文档文本
    :param method: 评分方法，可选值：keyword, position, semantic, hybrid
    :return: 相关性分数 (0.0-1.0)
    """
    query_lower = query.lower()
    doc_lower = document.lower()
    
    # 1. 关键词匹配评分
    if method == "keyword" or method == "hybrid":
        query_words = set(re.findall(r'[\w\u4e00-\u9fa5]+', query_lower))
        doc_words = set(re.findall(r'[\w\u4e00-\u9fa5]+', doc_lower))
        
        if not query_words:
            keyword_score = 0.0
        else:
            # 计算精确匹配
            exact_matches = query_words.intersection(doc_words)
            keyword_score = len(exact_matches) / len(query_words)
            
            # 计算部分匹配（前缀匹配）
            partial_matches = 0
            for q_word in query_words:
                if q_word not in exact_matches:
                    for d_word in doc_words:
                        if d_word.startswith(q_word) or q_word.startswith(d_word):
                            partial_matches += 0.5  # 部分匹配权重较低
                            break
            
            # 调整关键词分数
            keyword_score = min(1.0, keyword_score + partial_matches / len(query_words))
    
    # 2. 位置加权评分（文档开头的匹配更重要）
    if method == "position" or method == "hybrid":
        # 计算查询词在文档中的位置权重
        position_score = 0.0
        doc_length = len(doc_lower)
        
        if doc_length > 0:
            query_terms = re.findall(r'[\w\u4e00-\u9fa5]+', query_lower)
            
            for term in query_terms:
                pos = doc_lower.find(term)
                if pos != -1:
                    # 位置越靠前，分数越高（使用对数缩放）
                    position_weight = 1.0 - (pos / (doc_length * 1.0))
                    position_score += position_weight
            
            if query_terms:
                position_score = position_score / len(query_terms)
    
    # 3. 语义相关性评分（简化版，实际应用中可以使用嵌入向量）
    if method == "semantic" or method == "hybrid":
        # 简化的语义评分：检查查询词是否在文档中出现，以及出现的频率
        semantic_score = 0.0
        query_terms = re.findall(r'[\w\u4e00-\u9fa5]+', query_lower)
        
        if query_terms:
            # 计算查询词的出现频率
            freq_sum = 0
            for term in query_terms:
                freq = doc_lower.count(term)
                # 频率越高，分数越高，但有上限
                freq_sum += min(3, freq)  # 最多计算3次出现
            
            # 归一化到0-1范围
            semantic_score = min(1.0, freq_sum / (len(query_terms) * 3.0))
    
    # 根据选择的方法返回分数
    if method == "keyword":
        return keyword_score
    elif method == "position":
        return position_score
    elif method == "semantic":
        return semantic_score
    elif method == "hybrid":
        # 混合评分：关键词匹配(40%) + 位置加权(30%) + 语义相关性(30%)
        hybrid_score = (keyword_score * 0.4) + (position_score * 0.3) + (semantic_score * 0.3)
        return min(1.0, hybrid_score)
    else:
        raise ValueError(f"不支持的评分方法: {method}")


def rerank_documents(query: str, retrieved_docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    对检索到的文档进行重排序，提升相关性
    
    :param query: 查询文本
    :param retrieved_docs: 初始检索到的文档列表
    :param top_k: 返回的文档数量
    :return: 重排序后的文档列表
    """
    if not retrieved_docs:
        return []
    
    # 计算每个文档的相关性分数
    scored_docs = []
    for item in retrieved_docs:
        doc = item['doc']
        # 使用混合评分方法
        relevance_score = calculate_relevance_score(query, doc.page_content, method="hybrid")
        
        # 合并原始分数和计算的相关性分数
        original_score = item.get('score', 0.0)
        
        # 智能加权：如果原始分数有效（>0），则结合两种分数
        if original_score > 0:
            # 归一化原始分数到0-1范围（假设原始分数最大约为1）
            normalized_original = min(1.0, original_score)
            # 综合评分：原始向量相似度(60%) + 文本相关性(40%)
            combined_score = (normalized_original * 0.6) + (relevance_score * 0.4)
        else:
            combined_score = relevance_score
        
        scored_docs.append({
            **item,
            "text_relevance": relevance_score,
            "combined_score": combined_score
        })
    
    # 按综合分数排序
    reranked_docs = sorted(scored_docs, key=lambda x: x.get('combined_score', 0), reverse=True)
    
    # 返回前top_k个文档
    result = reranked_docs[:top_k]
    logger.info(f"文档重排序完成，从{len(retrieved_docs)}个文档中选择{len(result)}个最相关的文档")
    
    return result


def extract_key_information(text: str, max_length: int = 200) -> str:
    """
    从文本中提取关键信息，用于摘要或快速预览
    
    :param text: 原始文本
    :param max_length: 最大长度
    :return: 提取的关键信息
    """
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 简单实现：提取前max_length个字符
    if len(text) <= max_length:
        return text
    
    # 尝试在句子边界截断
    truncate_pos = max_length
    for i in range(max_length, max(0, max_length - 50), -1):
        if text[i:i+1] in ['。', '！', '？', '.', '!', '?', ';', '；']:
            truncate_pos = i + 1
            break
    
    return text[:truncate_pos] + "..."
