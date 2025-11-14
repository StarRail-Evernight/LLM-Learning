from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch

embedding = HuggingFaceEmbeddings(
    model_name="D:/LLM_download/BAAI/bge-small-zh-v1.5",  # 这里用了BBAI的词嵌入模型，可以在HuggingFace/魔搭社区上找到
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
