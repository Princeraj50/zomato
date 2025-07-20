import streamlit as st
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

# 🌱 Load .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 🧠 Model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# 📦 State Schema
class PromptState(TypedDict):
    messages: List[BaseMessage]
    latest_prompt: str

# 🔧 Prompt refinement node
def refine_prompt(state: PromptState) -> PromptState:
    prompt = state["messages"][-1].content
    clarity_check = llm.invoke([
        HumanMessage(content=(
            f"Evaluate this prompt:\n'{prompt}'\n"
            "If it's vague, assume context and enhance it. If it's clear, refine it further for specificity and utility."
        ))
    ])
    improved_prompt = clarity_check.content
    optimized = llm.invoke([
        HumanMessage(content=(
            f"Rewrite the instruction clearly and naturally for a general-purpose AI assistant:\n\n{improved_prompt}"
        ))
    ])
    return {
        "messages": [AIMessage(content=optimized.content)],
        "latest_prompt": optimized.content
    }

# 🧠 Criticizer
def critique_optimizer_output(state: PromptState) -> PromptState:
    _ = llm.invoke([
        HumanMessage(content=(
            f"Critique this optimized instruction:\n\n'{state['latest_prompt']}'\n"
            "Improve wording internally if needed."
        ))
    ])
    return state

# ✅ Final Agent
def final_agent(state: PromptState) -> PromptState:
    final_prompt = state["latest_prompt"]
    suggestion = llm.invoke([
        HumanMessage(content=(
            f"The final enhanced prompt is:\n'{final_prompt}'\n"
            "Suggest one optional improvement (e.g. add timeframe, target format, goal)."
        ))
    ])
    final_output = (
        f"✅ Final optimized prompt:\n\n{final_prompt}\n\n"
        f"💡 Suggestion to improve even further:\n{suggestion.content}"
    )
    return {
        "messages": [AIMessage(content=final_output)],
        "latest_prompt": final_prompt
    }

# 🧱 Graph Setup
graph = StateGraph(PromptState)
for i in range(1, 4):
    graph.add_node(f"refine_{i}", refine_prompt)
    graph.add_node(f"criticize_{i}", critique_optimizer_output)
graph.add_node("final_agent", final_agent)

# 🔗 Edges
graph.set_entry_point("refine_1")
graph.add_edge("refine_1", "criticize_1")
graph.add_edge("criticize_1", "refine_2")
graph.add_edge("refine_2", "criticize_2")
graph.add_edge("criticize_2", "refine_3")
graph.add_edge("refine_3", "criticize_3")
graph.add_edge("criticize_3", "final_agent")
graph.add_edge("final_agent", END)

# 🚀 Compile
flow = graph.compile()

# 🖥️ Streamlit UI
st.title("🔧 Prompt Optimizer with LangGraph")
user_prompt = st.text_area("Enter your raw prompt:", placeholder="e.g., Who is the current president of India?")

if st.button("Optimize Prompt"):
    if user_prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        result = flow.invoke({
            "messages": [HumanMessage(content=user_prompt)],
            "latest_prompt": user_prompt
        })
        final_output = result["messages"][-1].content
        st.success("Optimization complete!")
        st.markdown(final_output)