import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from typing import TypedDict, Optional

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import openai
import os

# 🔐 Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# 🤖 GPT model
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# 📦 State schema
class ChatState(TypedDict):
    query: str
    response: str
    image_prompt: Optional[str]

# 🍽️ Reply generator
def generate_reply(state: ChatState) -> ChatState:
    user_msg = state["query"]
    system_prompt = (
        "You're a hyper-friendly foodie chatbot helping users choose meals from Zomato. "
        "Use vivid language, emojis, and always end with an image prompt like:\nIMAGE: sizzling paneer tikka with mint chutney"
    )
    messages = [HumanMessage(content=f"{system_prompt}\n{user_msg}")]
    response = llm(messages).content

    # Extract image prompt
    image_prompt = None
    if "IMAGE:" in response:
        image_prompt = response.split("IMAGE:")[-1].strip().split("\n")[0]

    return {
        "query": user_msg,
        "response": response.replace("Bot:", "").strip(),
        "image_prompt": image_prompt
    }

# 🎨 Generate image
def generate_image(prompt: str) -> Optional[str]:
    try:
        res = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1
        )
        return res.data[0].url
    except Exception as e:
        st.warning(f"Image generation failed: {e}")
        return None

# 🧠 LangGraph setup
graph = StateGraph(ChatState)
graph.add_node("chatbot", generate_reply)
graph.set_entry_point("chatbot")
flow = graph.compile()

# 🎮 UI Config
st.set_page_config(page_title="Foodie Chatbot 🍕", layout="centered")
st.title("😋 Your Foodie Buddy")
st.subheader("Tempting suggestions, tasty chats.")

# 🔄 Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🗣️ Continuous input field
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("🍽️ Type your craving...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    result = flow.invoke({"query": user_input, "response": "", "image_prompt": None})
    image_url = generate_image(result["image_prompt"]) if result["image_prompt"] else None

    st.session_state.chat_history.append({
        "role": "user", "content": user_input
    })
    st.session_state.chat_history.append({
        "role": "assistant", "content": result["response"], "image_url": image_url
    })

# 💬 Chat display
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(msg["content"])
        if msg.get("image_url"):
            st.image(msg["image_url"], caption="🔥 Craving visualized")