# -*- coding: utf-8 -*-
# @time  : 2025/11/14 21:26
# @author: 'March 7th'
# @gitee : 'https://gitee.com/StarRail-Evernight'

from langchain_community.chat_models import ChatZhipuAI
from langchain_ollama import ChatOllama


def get_zhipu(model: str | None = "glm-4.6", temperature: float | None = 0.5) -> ChatZhipuAI:
    """
    获取智普AI模型
    :param model: 模型名称，默认"glm-4.6"
    :param temperature: 温度参数，默认0.5
    :return: ChatZhipuAI模型实例
    """
    return ChatZhipuAI(
        model=model,
        temperature=temperature,
    )


def get_ollama(model: str | None = "qwen3:4b_q4_k_m", base_url: str | None = "http://localhost:10086", temperature: float | None = 0.8) -> ChatOllama:
    """
    获取Ollama模型
    :param model: 模型名称，默认"qwen3:4b_q4_k_m"
    :param base_url: Ollama服务地址，默认"http://localhost:10086"
    :param temperature: 温度参数，默认1.0
    :return: ChatOllama模型实例
    """
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


if __name__ == '__main__':
    llm = get_zhipu()
    response = llm.invoke("你好")
    print(response.content)
