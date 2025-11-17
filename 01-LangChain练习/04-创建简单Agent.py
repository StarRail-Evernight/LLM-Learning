from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
import requests
import os


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
    except:
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


llm = ChatOllama(model="qwen3:4b_q4_k_m", temperature=1., base_url="http://localhost:11434")

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="你是一个专业的天气查询助手，你可以回答用户的问题。",

)

if __name__ == '__main__':
    user_query = "重庆未来3天天气如何？"

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_query}]
    })

    print(result["messages"][-1].content)
