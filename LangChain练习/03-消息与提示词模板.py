from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # 消息类型
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate  # 提示词模板

# 消息类型,这里只是简单的展示一下消息的类型
system_message = SystemMessage(content="你是一个专业的翻译助手，你会将用户输入的中文翻译成英文")  # 系统消息，用于设置模型的行为
human_message = HumanMessage(content="你好")  # 用户提问
ai_message = AIMessage(content="Hello")  # 模型回答

# print(human_message.content)
# print(ai_message.content)
# print(system_message.content)

# 提示词模板
template = PromptTemplate.from_template("请将{input}翻译成英文")  # 从模板创建一个提示词模板

result = template.format(input="你好")  # 填充内容
# print(result)
# print(template.invoke({"input": "你好"}))  # text='请将你好翻译成英文'

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个专业的翻译助手，你会将用户输入的中文翻译成英文"),  # 系统消息，用于设置模型的行为
        ("human", "{input}"),  # 用户提问
    ]
)
print(chat_prompt.invoke({"input": "你好"}))  # text='请将你好翻译成英文'
"""
messages=[SystemMessage(content='你是一个专业的翻译助手，你会将用户输入的中文翻译成英文', 
additional_kwargs={}, response_metadata={}), HumanMessage(content='你好', additional_kwargs={}, response_metadata={})]

"""
