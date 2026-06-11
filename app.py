import os
import streamlit as st
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

st.title("Multi-Agent AI Assistant")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)


class AgentState(TypedDict):
    user_question: str
    research_answer: str
    final_answer: str
    reviewed_answer: str


def research_agent(state: AgentState):
    prompt = f"""
You are a Research Agent.
Understand the user's question and collect the key points.

User question:
{state["user_question"]}
"""
    response = llm.invoke(prompt)
    return {"research_answer": response.content}


def writer_agent(state: AgentState):
    prompt = f"""
You are a Writer Agent.
Use the research below and write a clear beginner-friendly answer.

Research:
{state["research_answer"]}
"""
    response = llm.invoke(prompt)
    return {"final_answer": response.content}


def reviewer_agent(state: AgentState):
    prompt = f"""
You are a Reviewer Agent.

Review and improve the answer below.

Improve:
- grammar
- clarity
- formatting
- readability
- remove repetition
- keep it under 250 words

Answer:
{state["final_answer"]}
"""
    response = llm.invoke(prompt)
    return {"reviewed_answer": response.content}


graph = StateGraph(AgentState)

graph.add_node("research_agent", research_agent)
graph.add_node("writer_agent", writer_agent)
graph.add_node("reviewer_agent", reviewer_agent)

graph.set_entry_point("research_agent")

graph.add_edge("research_agent", "writer_agent")
graph.add_edge("writer_agent", "reviewer_agent")
graph.add_edge("reviewer_agent", END)

app = graph.compile()

user_input = st.text_input("Ask something")

if st.button("Submit"):
    if user_input:
        result = app.invoke({"user_question": user_input})

        st.subheader("Research Agent Output")
        st.write(result["research_answer"])

        st.subheader("Writer Agent Output")
        st.write(result["final_answer"])

        st.subheader("Reviewer Agent Final Answer")
        st.write(result["reviewed_answer"])
    else:
        st.warning("Please enter a question.")