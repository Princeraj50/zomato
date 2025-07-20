import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from typing import TypedDict

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os

# 🔐 Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 🤖 Set up GPT chatbot
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# 📦 LangGraph state schema
class ChatState(TypedDict):
    query: str
    response: str

# 🧠 LangGraph node: friendly food reply
def generate_reply(state: ChatState) -> ChatState:
    user_msg = state["query"]
    system_prompt = (
        "You're a hyper-friendly foodie chatbot that helps people pick the tastiest meals from Zomato. "
        "You LOVE food. Be playful, witty, and persuasive. Suggest dishes with vivid descriptions, emojis, and image links."
    )
    messages = [HumanMessage(content=f"{system_prompt}\n{user_msg}")]
    response = llm(messages)
    return {"query": user_msg, "response": response.content}

# 🧩 LangGraph setup
graph = StateGraph(ChatState)
graph.add_node("chatbot", generate_reply)
graph.set_entry_point("chatbot")
flow = graph.compile()

# 🍽️ Streamlit UI
st.set_page_config(page_title="Foodie Chatbot 🍕", layout="centered")
st.title("😋 Your Foodie Buddy")
st.subheader("Let’s spice things up! Cravings, delivered.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🖼️ Extract image URL from bot reply
def extract_image_url(text: str) -> str | None:
    for word in text.split():
        if word.startswith("http") and ("jpg" in word or "png" in word):
            return word.strip()
    return None

# 🗣️ User input
user_input = st.text_input("🍔 What's making your stomach growl?")
if st.button("Send") and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    result = flow.invoke({"query": user_input})
    bot_reply = result["response"]
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

# 💬 Display conversation
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
        image_url = extract_image_url(msg["content"])
        if image_url:
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Feeling hungry yet? 😍")
            except:
                st.warning("Couldn't load image — but trust me, it was drool-worthy 😋")