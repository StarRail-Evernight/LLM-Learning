import os
import requests
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  # 内存保存器，用于保存对话历史


llm = ChatOllama(
    model="qwen3:4b_q4_k_m",
    temperature=0.5,
    base_url="http://localhost:11434"
)


@tool
def get_weather(location: str, days: int) -> str:
    """
    获取对应城市天气信息
    :param location: 城市位名称
    :param days: 天数信息，若为1，则只查询今天的信息
    :return: 获取对应城市天气信息
    """
    try:
        weather_inf = requests.get(
            url=f"https://api.seniverse.com/v3/weather/daily.json?"
                f"key={os.getenv('XIN_ZHI_API_KEY')}&location={location}&language=zh-Hans&unit=c&start=0&days={days}"
        )
    except requests.RequestException:
        return "获取天气信息失败"

    results = ""
    for data in weather_inf.json()["results"][0]["daily"]:
        if data["text_day"] == data["text_night"]:
            results += (
                f"日期: {data['date']}\n"
                f"天气状况: {data['text_day']}\n"
                f"温度: {data['low']}~{data['high']}℃\n\n"
            )
        else:
            results += (
                f"日期: {data['date']}\n"
                f"天气状况: {data['text_day']}转{data['text_night']}\n"
                f"温度: {data['low']}~{data['high']}℃\n\n"
            )

    return results


agent = create_agent(
    llm,
    [get_weather],
    checkpointer=InMemorySaver(),  # 内存保存器，用于保存对话历史
)

if __name__ == '__main__':
    while True:
        user_input = input("请输入您的问题：")
        if user_input == "exit":
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": "1"}}  # 线程ID，用于区分不同的对话
        )
        print(result["messages"][-1].content)

"""
请输入您的问题：重庆今天天气如何？
今天重庆的天气晴朗，气温在7℃到19℃之间，适合外出活动。
请输入您的问题：我之前问了什么问题？
您之前询问的是“重庆今天天气如何？”。我为您查询了重庆今天的天气信息，并给出了晴朗、温度7~19℃的答复。
请输入您的问题：exit

请输入您的问题：重庆未来三天天气如何？
根据查询结果，重庆未来三天的天气预报如下：

📅 2025年11月28日：晴天，气温7℃~19℃
📅 2025年11月29日：晴天，气温8℃~17℃
📅 2025年11月30日：晴转多云，气温9℃~18℃

天气整体以晴朗为主，气温逐渐回升，建议根据温度变化适当增添衣物。需要其他天气信息可以随时告诉我哦~
请输入您的问题：根据查询到的天气信息给出穿衣建议
根据重庆未来三天的天气预报，我为您整理穿衣建议如下：

🧣 **11月28日（晴）**  
早间气温7℃，建议穿厚外套+毛衣，下午19℃可换薄卫衣；注意早晚温差大，可携带薄外套备用。

🌤️ **11月29日（晴）**  
气温8~17℃，适合穿针织衫+长裤，早晚可加薄外套，白天适合轻便外套或风衣。

🌤️ **11月30日（晴转多云）**  
气温9~18℃，建议穿衬衫+薄外套，早晚多云可能稍凉，可备一件针织开衫。

💡 **温馨提示**  
3天整体气温逐渐回升，但早晚仍较凉，建议根据实时温度调整衣物，注意保暖防风哦~ 🌬️
请输入您的问题：exit
"""

"""
在生产环境中，请使用由数据库支持的 checkpointer：
安装PostgreSQL数据库 pip install langgraph-checkpoint-postgres


from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # 在PostgresSQL中自动创建表格
    agent = create_agent(
        "openai:gpt-5",
        [get_user_info],
        checkpointer=checkpointer,
    )
"""
