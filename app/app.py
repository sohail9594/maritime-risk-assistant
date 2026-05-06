"""Streamlit entry point for the maritime AI chatbot."""

import streamlit as st

from utils.rag_pipeline import answer_question
from utils.routing import suggest_alternative_route


st.set_page_config(page_title="Maritime AI Chatbot", page_icon=":ship:")

st.title("Maritime AI Chatbot")
st.write("Ask questions about maritime documents and get simple route suggestions.")

question = st.text_input("Ask a question")

if question:
    st.subheader("Answer")
    st.write(answer_question(question))

st.divider()

st.subheader("Route Suggestion")
origin = st.text_input("Origin port")
destination = st.text_input("Destination port")

if st.button("Suggest route"):
    st.write(suggest_alternative_route(origin, destination))
