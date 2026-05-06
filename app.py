"""Streamlit frontend for the local maritime RAG chatbot."""

from pathlib import Path

import streamlit as st

from utils.decision_logic import (
    is_fragile_electronics_transport_query,
    is_iot_satellite_jamming_query,
    is_medical_diversion_decision_query,
)
from utils.logistics import is_trucking_time_query
from utils.rag_pipeline import (
    answer_question,
    build_rag_pipeline,
    is_reference_priority_query,
)
from utils.routing import is_route_query


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_ROOT / "data"

EXAMPLE_QUESTIONS = [
    "Is the Strait of Hormuz safe?",
    "Suggest a safer route from Dubai to Mumbai",
    "How long from Salalah to Dubai by truck?",
    "Should I divert insulin from Jebel Ali to Salalah?",
    "Good idea to truck AI server racks through an unpaved mountain pass?",
]


def count_documents() -> int:
    """Count local .txt documents available to the RAG pipeline."""
    if not DATA_FOLDER.exists():
        return 0

    return len(list(DATA_FOLDER.glob("*.txt")))


def should_use_rag(query: str) -> bool:
    """Return True when the query should go to the document retriever."""
    if is_reference_priority_query(query):
        return True

    return not (
        is_fragile_electronics_transport_query(query)
        or is_iot_satellite_jamming_query(query)
        or is_medical_diversion_decision_query(query)
        or is_route_query(query)
        or is_trucking_time_query(query)
    )


@st.cache_resource(show_spinner=False)
def get_cached_retriever():
    """Build the local RAG retriever once and reuse it across chat turns."""
    return build_rag_pipeline()


def get_answer(query: str) -> str:
    """Call the existing backend answer function."""
    if should_use_rag(query):
        retriever = get_cached_retriever()
        if retriever is None:
            return "I don't have enough data."

        return answer_question(query, retriever)

    return answer_question(query)


def format_chat_history(messages: list[dict[str, str]]) -> str:
    """Convert chat messages into a simple text transcript."""
    if not messages:
        return "No chat history yet."

    transcript_lines = ["Maritime Risk & Rerouting Assistant - Chat History", ""]

    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        transcript_lines.append(f"{role}:")
        transcript_lines.append(message["content"])
        transcript_lines.append("")

    return "\n".join(transcript_lines).strip() + "\n"


st.set_page_config(
    page_title="Maritime Risk & Rerouting Assistant",
    page_icon=":ship:",
    layout="centered",
)

st.title("Maritime Risk & Rerouting Assistant")
st.write(
    "A free local assistant that combines document-based RAG with simple "
    "rule-based routing, trucking-time, and medical logistics decision logic."
)

with st.sidebar:
    st.header("Project Features")
    st.markdown(
        "- Local RAG over maritime text documents\n"
        "- ChromaDB vector search\n"
        "- Sentence Transformers embeddings\n"
        "- Rule-based route suggestions\n"
        "- Logistics and medical cargo decision handling"
    )

    st.header("Dataset")
    st.metric("Text documents", count_documents())

    st.header("Example Questions")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.sidebar:
    st.download_button(
        "Download Chat History",
        data=format_chat_history(st.session_state.messages),
        file_name="maritime_chat_history.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed_question = st.chat_input("Describe a maritime or logistics issue...")
question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing maritime documents..."):
            answer = get_answer(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
