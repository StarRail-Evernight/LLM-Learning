from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatZhipuAI(
    model="glm-4.6",
    temperature=0.5
)

message = [
    SystemMessage(content="你是一个专业的翻译助手，你会将用户输入的中文翻译成英文"),  # 系统消息，用于设置模型的行为
    HumanMessage(content="大语言模型")  # 用户提问
]

result = llm.invoke(message)
print(result.content)
