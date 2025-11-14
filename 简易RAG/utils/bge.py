from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # 新版本LangChain改成这样了
import torch


def get_embedding(path: str | None = None) -> HuggingFaceEmbeddings:
    """
    获取BGE模型的词嵌入模型
    :param path: 词嵌入模型路径，默认使用"D:/LLM_download/BAAI/bge-small-zh-v1.5"
    :return: HuggingFaceEmbeddings 词嵌入模型
    """
    return HuggingFaceEmbeddings(
        model_name=path or "D:/LLM_download/BAAI/bge-small-zh-v1.5",  # 这里用了BBAI的词嵌入模型，可以在HuggingFace/魔搭社区上找到
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


if __name__ == '__main__':
    embedding = get_embedding()
    # 测试一下
    test_sentence = "这是一个测试句子"
    test_embedding = embedding.embed_query(test_sentence)
    print(test_embedding[:5])  # 打印前5个维度的嵌入向量
