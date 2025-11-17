from langchain_community.chat_models import ChatZhipuAI  # 调用带有API_Key的模型
from langchain_ollama import ChatOllama  # 调用本地模型，这里的Ollama需要安装ollama

zhipu = ChatZhipuAI(
    model="glm-4.6",
    temperature=0.5,
    # api_key="sk-xxxx", # 填写API Key, 可以在环境变量中配置，配置后可以不填写
)

qwen = ChatOllama(
    model="qwen3:4b_q4_k_m",  # 这里的模型需要在ollama中提前pull下来，也可以量化之后使用
    temperature=0.5,
    base_url="http://localhost:11434"  # 填写Ollama的地址，默认是http://localhost:11434
)

# ================ 简单调用 ================
# result = zhipu.invoke("你好")
# print(result.content)
# print("=" * 20)
# result = qwen.invoke("你好")
# print(result.content)


# ================ 流式输出 (带chunk的均与流式输出有关) ================
# result = zhipu.stream("你好")
# for chunk in result:
#     print(chunk.content, end="")


# ================ 异步调用 ================
# import asyncio
#
#
# async def main():
#     result = await qwen.ainvoke("你好")  # 这里的异步调用需要用ainvoke 流式输出用astream
#     print(result.content)
#
#
# asyncio.run(main())


llm = zhipu.with_fallbacks([qwen])  # 在zhipu不可用的情况下，会自动调用qwen

result = llm.invoke("你好")
print(result.content)
