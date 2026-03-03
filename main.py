import uuid
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from backend.core import chatbot, retrieve_all_threads, ingest_pdf, thread_document_metadata

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Sidebar thread buttons */
        div[data-testid="stSidebarContent"] .stButton > button {
            width: 100%;
            text-align: left;
            font-size: 0.8rem;
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        /* Active thread highlight */
        div[data-testid="stSidebarContent"] .stButton > button:focus {
            border-color: #4a90d9;
            background-color: #1e3a5f22;
        }
        /* Slightly tighten chat bubbles */
        .stChatMessage { padding: 0.6rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def new_thread_id() -> str:
    return str(uuid.uuid4())


def thread_label(thread_id: str, index: int) -> str:
    """Human-readable sidebar label for a thread."""
    short = thread_id[:8]
    return f"💬 Chat {index + 1}  ·  {short}…"


def load_conversation(thread_id: str) -> list[dict]:
    """Fetch LangGraph state and convert to [{role, content}] dicts."""
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        raw_messages = state.values.get("messages", [])
    except Exception as exc:
        st.error(f"Could not load conversation: {exc}")
        return []

    history = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            history.append({"role": "assistant", "content": msg.content})
        # ToolMessages are intentionally skipped for the display history
    return history


def reset_chat() -> None:
    """Start a brand-new conversation thread."""
    tid = new_thread_id()
    st.session_state.thread_id = tid
    st.session_state.message_history = []
    _register_thread(tid)


def _register_thread(thread_id: str) -> None:
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.insert(0, thread_id)  # newest first


def switch_thread(thread_id: str) -> None:
    """Load an existing thread into the active view."""
    st.session_state.thread_id = thread_id
    st.session_state.message_history = load_conversation(thread_id)


def stream_ai_response(user_input: str, config: dict):
    """
    Generator: streams AIMessage chunks, shows status boxes for tool calls.
    Yields text chunks for st.write_stream.
    """
    status_box = None
    pending_tool_name = None
    interrupt_value = None

    for message_chunk, _metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(message_chunk, ToolMessage):
            tool_name = getattr(message_chunk, "name", "tool")
            if status_box is None:
                status_box = st.status(f"🔧 Using `{tool_name}`…", expanded=True)
            else:
                status_box.update(
                    label=f"🔧 Using `{tool_name}`…",
                    state="running",
                    expanded=True,
                )
            pending_tool_name = tool_name

        elif isinstance(message_chunk, AIMessage) and message_chunk.content:
            # Close tool status box once AI starts responding
            if status_box is not None and pending_tool_name:
                status_box.update(
                    label=f"✅ `{pending_tool_name}` finished",
                    state="complete",
                    expanded=False,
                )
                status_box = None
                pending_tool_name = None
            yield message_chunk.content

    # Edge case: tool was the last message (no AI text after)
    if status_box is not None:
        status_box.update(
            label="✅ Tool finished",
            state="complete",
            expanded=False,
        )
    state = chatbot.get_state(config=config)
    if state.next:  # graph is paused / waiting
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_msg = task.interrupts[0].value  # e.g. "Approve buying 5 shares of AAPL?"
                st.session_state.pending_interrupt = {
                    "config": config,
                    "message": interrupt_msg,
                }
                break

# ──────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ──────────────────────────────────────────────────────────────────────────────

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads() or []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

# RAG: per-thread ingested doc tracking  { thread_id: { filename: summary_dict } }
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = {}

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

# Make sure the active thread is registered
_register_thread(st.session_state.thread_id)

# Convenience refs
thread_key = st.session_state.thread_id
thread_docs: dict = st.session_state.ingested_docs.setdefault(thread_key, {})

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 LangGraph Chat")
    st.divider()

    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()

    # ── RAG: PDF upload section ──────────────────────────────────────────────
    st.subheader("📄 Document (RAG)")

    # Show currently indexed doc for this thread
    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.success(
            f"**{latest_doc.get('filename')}**\n\n"
            f"{latest_doc.get('chunks')} chunks · {latest_doc.get('documents')} pages"
        )
    else:
        st.info("No PDF indexed for this chat yet.")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.info(f"`{uploaded_pdf.name}` is already indexed for this chat.")
        else:
            with st.status("Indexing PDF…", expanded=True) as status_box:
                try:
                    summary = ingest_pdf(
                        uploaded_pdf.getvalue(),
                        thread_id=thread_key,
                        filename=uploaded_pdf.name,
                    )
                    thread_docs[uploaded_pdf.name] = summary
                    # Keep ingested_docs in sync
                    st.session_state.ingested_docs[thread_key] = thread_docs
                    status_box.update(
                        label="✅ PDF indexed", state="complete", expanded=False
                    )
                except Exception as exc:
                    status_box.update(
                        label=f"❌ Indexing failed: {exc}",
                        state="error",
                        expanded=True,
                    )

    st.divider()

    # ── Conversation history ─────────────────────────────────────────────────
    st.subheader("Conversations")

    threads = st.session_state.chat_threads
    if not threads:
        st.caption("No conversations yet.")
    else:
        for idx, tid in enumerate(threads):
            label = thread_label(tid, len(threads) - 1 - idx)
            is_active = tid == st.session_state.thread_id
            btn_type = "secondary" if is_active else "tertiary"
            if st.button(
                label,
                key=f"thread_{tid}",
                use_container_width=True,
                type=btn_type,
            ):
                switch_thread(tid)
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────────────────────────────────────

st.header("Chat", divider="gray")

# Show active doc badge in main area if a PDF is indexed
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.caption(
        f"📎 RAG active — **{latest_doc.get('filename')}** "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )

# Render existing messages
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask about your document or chat freely…")

# ── Human-in-the-loop: Stock Purchase Approval ────────────────────────────────
if "pending_interrupt" in st.session_state and st.session_state.pending_interrupt:
    interrupt_data = st.session_state.pending_interrupt

    with st.chat_message("assistant"):
        st.warning(f"⚠️ **Approval Required**\n\n{interrupt_data['message']}")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Yes, approve", type="primary", use_container_width=True):
                st.session_state.pending_interrupt = None

                # Resume the graph with "yes"
                with st.chat_message("assistant"):
                    with st.spinner("Processing..."):
                        result = chatbot.invoke(
                            Command(resume="yes"),
                            config=interrupt_data["config"],
                        )
                    # Get last AI message
                    last_ai = next(
                        (m for m in reversed(result["messages"])
                         if isinstance(m, AIMessage) and m.content),
                        None
                    )
                    if last_ai:
                        st.markdown(last_ai.content)
                        st.session_state.message_history.append(
                            {"role": "assistant", "content": last_ai.content}
                        )
                st.rerun()

        with col2:
            if st.button("❌ No, cancel", use_container_width=True):
                st.session_state.pending_interrupt = None

                with st.chat_message("assistant"):
                    with st.spinner("Processing..."):
                        result = chatbot.invoke(
                            Command(resume="no"),
                            config=interrupt_data["config"],
                        )
                    last_ai = next(
                        (m for m in reversed(result["messages"])
                         if isinstance(m, AIMessage) and m.content),
                        None
                    )
                    if last_ai:
                        st.markdown(last_ai.content)
                        st.session_state.message_history.append(
                            {"role": "assistant", "content": last_ai.content}
                        )
                st.rerun()

if user_input:
    # Show user message immediately
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "metadata": {"thread_id": st.session_state.thread_id},
        "run_name": "chat_turn",
    }

    # Stream assistant response
    with st.chat_message("assistant"):
        try:
            ai_response = st.write_stream(stream_ai_response(user_input, config))
        except Exception as exc:
            st.error(f"An error occurred while getting a response: {exc}")
            ai_response = None

    if ai_response:
        st.session_state.message_history.append(
            {"role": "assistant", "content": ai_response}
        )

        # Refresh doc metadata from backend (in case RAG tool updated it)
        try:
            doc_meta = thread_document_metadata(thread_key)
            if doc_meta and doc_meta.get("filename") not in thread_docs:
                thread_docs[doc_meta["filename"]] = doc_meta
                st.session_state.ingested_docs[thread_key] = thread_docs
        except Exception:
            pass  # non-critical