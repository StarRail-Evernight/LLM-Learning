import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# 设置页面配置
st.set_page_config(
    page_title="Qwen3 本地聊天助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面标题
st.title("🤖 Qwen3 本地聊天助手")
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 模型配置")

    # 模型参数设置
    model_name = st.selectbox(
        "选择模型",
        options=["qwen3:4b_q4_k_m", "deepseek-r1:8b", "qwen3:8b"],
        index=0,
        help="需要在本地Ollama中已下载的模型"
    )
    temperature = st.slider(
        "温度系数",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="控制输出的随机性，值越高越随机，越低越确定"
    )

    base_url = st.text_input(
        "Ollama服务地址",
        value="http://localhost:11434",
        help="本地Ollama服务的地址和端口"
    )

    st.markdown("---")
    st.info(
        "📋 使用说明：\n"
        "1. 确保本地Ollama服务已启动（ollama serve）\n"
        "2. 已下载对应模型（ollama pull 模型名）\n"
        "3. 在输入框中输入问题并发送"
    )

# 初始化对话历史（使用session state持久化）
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化模型（使用session state避免重复创建）
@st.cache_resource(show_spinner="正在初始化模型...")
def init_model(model, temp, url):
    try:
        return ChatOllama(
            model=model,
            temperature=temp,
            base_url=url,
        )
    except Exception as e:
        st.error(f"模型初始化失败：{str(e)}")
        return None

# 初始化模型
model = init_model(model_name, temperature, base_url)

# 显示对话历史
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 聊天输入框
if prompt := st.chat_input("请输入你的问题..."):
    # 检查模型是否初始化成功
    if model is None:
        st.error("模型未初始化成功，请检查配置和Ollama服务状态！")
    else:
        # 添加用户消息到对话历史
        st.session_state.messages.append(HumanMessage(content=prompt))

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成助手回复
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # 流式获取回复（模拟打字效果）
                response = model.stream([HumanMessage(content=prompt)])

                for chunk in response:
                    if chunk.content:
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")

                # 显示完整回复
                message_placeholder.markdown(full_response)

                # 添加助手消息到对话历史
                st.session_state.messages.append(AIMessage(content=full_response))

            except Exception as e:
                error_msg = f"请求失败：{str(e)}"
                message_placeholder.markdown(f"❌ {error_msg}")
                st.error(error_msg)

# 清除对话历史按钮（修复列配置错误）
if st.session_state.messages:
    # 只使用两个列，比例之和为1.0，都是正数
    col1, col2 = st.columns([0.9, 0.1])
    with col2:
        if st.button("🗑️ 清除历史"):
            st.session_state.messages = []
            st.rerun()