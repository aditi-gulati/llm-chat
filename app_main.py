"""
💬 Unified LLM Chat Application
Support for Ollama, OpenAI (GPT-4, GPT-3.5), and Anthropic (Claude Sonnet, Haiku)
Real-time streaming, advanced settings, chat history & export
File upload support for documents, PDFs, images, and text files
"""

import streamlit as st
import json
from datetime import datetime
import time
from typing import List, Dict
import os
from dotenv import load_dotenv
import base64
from io import StringIO

# Load environment variables
load_dotenv()

# Import unified LLM manager
from llm_provider import get_llm_manager

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="💬 Unified LLM Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 2rem; }
    .stChatMessage { border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0; }
    .response-time { font-size: 0.8rem; color: gray; margin-top: 0.5rem; }
    .provider-badge { padding: 0.3rem 0.6rem; background-color: #e8f4f8; border-radius: 0.3rem; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_provider" not in st.session_state:
    st.session_state.current_provider = None

if "current_model" not in st.session_state:
    st.session_state.current_model = None

if "settings" not in st.session_state:
    st.session_state.settings = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "num_predict": 512,
        "num_ctx": 2048,
    }

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "file_contents" not in st.session_state:
    st.session_state.file_contents = {}

# ============================================================================
# INITIALIZE LLM MANAGER
# ============================================================================

llm_manager = get_llm_manager()
available_providers = llm_manager.get_available_providers()

if not available_providers:
    st.error("❌ No LLM providers configured. Please set API keys in .env file")
    st.stop()

# ============================================================================
# FILE PROCESSING UTILITIES
# ============================================================================

def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file"""
    try:
        file_type = uploaded_file.type
        filename = uploaded_file.name

        # Text files
        if file_type in ["text/plain", "text/markdown"]:
            text = uploaded_file.getvalue().decode("utf-8")
            return text

        # CSV files
        elif file_type == "text/csv":
            text = uploaded_file.getvalue().decode("utf-8")
            return f"CSV File Content:\n{text}"

        # JSON files
        elif file_type == "application/json":
            text = uploaded_file.getvalue().decode("utf-8")
            return f"JSON File Content:\n{text}"

        # PDF files (basic extraction)
        elif file_type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return f"PDF Content:\n{text[:3000]}..."  # Limit to first 3000 chars
            except ImportError:
                return "⚠️ PDF support requires PyPDF2. Install with: pip install PyPDF2"

        # Image files
        elif file_type.startswith("image/"):
            return f"🖼️ Image file: {filename}\n(Image analysis requires vision models. Use image description in your query.)"

        # Word documents
        elif file_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            try:
                from docx import Document
                doc = Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])
                return f"Word Document Content:\n{text[:3000]}..."
            except ImportError:
                return "⚠️ Word support requires python-docx. Install with: pip install python-docx"

        else:
            return f"⚠️ File type '{file_type}' not directly supported. Attempting text extraction..."

    except Exception as e:
        return f"❌ Error processing file: {str(e)}"

def get_file_summary(content: str) -> str:
    """Generate a brief summary of file content"""
    lines = content.split('\n')
    preview = '\n'.join(lines[:5])
    char_count = len(content)
    return f"📄 **File Summary:**\n- Characters: {char_count}\n- Preview:\n```\n{preview}\n```"

# ============================================================================
# HEADER
# ============================================================================

st.markdown("# 💬 Unified LLM Chat")
st.markdown("*Support for Ollama, OpenAI (GPT-4, GPT-3.5), and Claude (Sonnet, Haiku)*")

# ============================================================================
# SIDEBAR - PROVIDER & MODEL SELECTION
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Provider & Model Selection")
    st.divider()

    # Provider Selection
    st.markdown("### 🔌 Select Provider")

    provider_options = {
        "ollama": "🦙 Ollama (Local/Remote)",
        "openai": "🟢 OpenAI (GPT-4, GPT-3.5)",
        "anthropic": "🧠 Anthropic (Claude)"
    }

    available_provider_options = {k: v for k, v in provider_options.items() if k in available_providers}

    if not available_provider_options:
        st.error("No providers available!")
        st.stop()

    selected_provider = st.selectbox(
        "Choose Provider:",
        options=list(available_provider_options.keys()),
        format_func=lambda x: available_provider_options[x],
        key="provider_select"
    )
    st.session_state.current_provider = selected_provider

    st.divider()

    # Model Selection
    st.markdown("### 🤖 Select Model")

    # Refresh Models Button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Load models from provider")
    with col2:
        if st.button("🔄 Refresh", help="Reload models from provider", use_container_width=True):
            st.rerun()

    models = llm_manager.get_models_for_provider(selected_provider)

    if models:
        selected_model = st.selectbox(
            "Available Models:",
            models,
            key="model_select"
        )
        st.session_state.current_model = selected_model

        # Show model info
        provider = llm_manager.get_provider(selected_provider)
        model_info = provider.get_model_info(selected_model)

        if model_info:
            with st.expander("📋 Model Details"):
                for key, value in model_info.items():
                    if key != "name":
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")

        st.success(f"✅ Selected: {selected_model}")
    else:
        st.warning(f"No models available for {selected_provider}")
        st.info("💡 For Ollama: Run `ollama pull mistral` in terminal\nFor OpenAI/Anthropic: Check API keys in .env")

    st.divider()

    # Advanced Settings
    st.markdown("### ⚡️ Advanced Settings")

    col1, col2 = st.columns(2)

    with col1:
        temp = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.settings["temperature"],
            step=0.1,
            help="0.0=Precise, 1.0=Balanced, 2.0=Creative"
        )
        st.session_state.settings["temperature"] = temp

        top_p = st.slider(
            "Top-P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.settings["top_p"],
            step=0.05
        )
        st.session_state.settings["top_p"] = top_p

    with col2:
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=100,
            value=st.session_state.settings["top_k"],
            step=1
        )
        st.session_state.settings["top_k"] = top_k

        num_predict = st.slider(
            "Max Response",
            min_value=128,
            max_value=4096,
            value=st.session_state.settings["num_predict"],
            step=128
        )
        st.session_state.settings["num_predict"] = num_predict

    st.divider()

    # File Upload
    st.markdown("### 📁 Upload Files")

    uploaded_files = st.file_uploader(
        "Upload documents (TXT, PDF, CSV, JSON, images, Word, etc.)",
        type=["txt", "pdf", "csv", "json", "md", "png", "jpg", "jpeg", "gif", "doc", "docx"],
        accept_multiple_files=True,
        help="Upload files to analyze with Ollama models"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

        for uploaded_file in uploaded_files:
            with st.expander(f"📄 {uploaded_file.name}"):
                file_content = extract_text_from_file(uploaded_file)

                # Store file content
                st.session_state.file_contents[uploaded_file.name] = file_content

                # Show preview
                if "Error" not in file_content and "⚠️" not in file_content:
                    st.write(get_file_summary(file_content))
                else:
                    st.info(file_content)

                # Option to use file in chat
                if st.button(f"💬 Use in Chat: {uploaded_file.name}", use_container_width=True, key=f"use_{uploaded_file.name}"):
                    # Add file context to next message
                    file_context = f"**[File: {uploaded_file.name}]**\n\n{file_content[:1000]}"
                    st.session_state.current_file = uploaded_file.name
                    st.info(f"✅ File '{uploaded_file.name}' will be included in your next query")

    st.divider()

    # Chat Management
    st.markdown("### 💾 Chat Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("📊 Stats", use_container_width=True):
            st.info(f"Messages: {len(st.session_state.messages)}")

    st.divider()

    st.markdown("### 📥 Export Chat")

    if st.session_state.messages:
        # Export format selector
        export_format = st.radio(
            "Select export format:",
            ["JSON", "TXT", "CSV", "Markdown"],
            horizontal=True,
            help="Choose format to export your chat"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Export", use_container_width=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                messages = st.session_state.messages
                provider = st.session_state.current_provider
                model = st.session_state.current_model

                if export_format == "JSON":
                    # Export as JSON
                    export_data = {
                        "provider": provider,
                        "model": model,
                        "timestamp": datetime.now().isoformat(),
                        "settings": st.session_state.settings,
                        "messages": messages
                    }
                    export_content = json.dumps(export_data, indent=2)
                    st.download_button(
                        "Download JSON",
                        export_content,
                        f"chat_{timestamp}.json",
                        "application/json",
                        use_container_width=True,
                        key="download_json"
                    )

                elif export_format == "TXT":
                    # Export as TXT
                    txt_content = f"CHAT EXPORT\n"
                    txt_content += f"{'='*60}\n"
                    txt_content += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    txt_content += f"Provider: {provider}\n"
                    txt_content += f"Model: {model}\n"
                    txt_content += f"{'='*60}\n\n"

                    for msg in messages:
                        role = msg.get("role", "unknown").upper()
                        content = msg.get("content", "")
                        txt_content += f"[{role}]\n{content}\n\n"

                    st.download_button(
                        "Download TXT",
                        txt_content,
                        f"chat_{timestamp}.txt",
                        "text/plain",
                        use_container_width=True,
                        key="download_txt"
                    )

                elif export_format == "CSV":
                    # Export as CSV
                    import csv
                    from io import StringIO

                    csv_buffer = StringIO()
                    csv_writer = csv.writer(csv_buffer)
                    csv_writer.writerow(["Timestamp", "Role", "Message", "Provider", "Model"])

                    for i, msg in enumerate(messages):
                        csv_writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            msg.get("role", "unknown"),
                            msg.get("content", "").replace("\n", " "),
                            provider if i == len(messages) - 1 else "",
                            model if i == len(messages) - 1 else ""
                        ])

                    st.download_button(
                        "Download CSV",
                        csv_buffer.getvalue(),
                        f"chat_{timestamp}.csv",
                        "text/csv",
                        use_container_width=True,
                        key="download_csv"
                    )

                elif export_format == "Markdown":
                    # Export as Markdown
                    md_content = f"# Chat Export\n\n"
                    md_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    md_content += f"**Provider:** `{provider}`\n\n"
                    md_content += f"**Model:** `{model}`\n\n"
                    md_content += f"**Total Messages:** {len(messages)}\n\n"
                    md_content += "---\n\n"

                    for msg in messages:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        if role == "user":
                            md_content += f"### You\n\n{content}\n\n"
                        else:
                            md_content += f"### Assistant\n\n{content}\n\n"
                        md_content += "---\n\n"

                    st.download_button(
                        "Download Markdown",
                        md_content,
                        f"chat_{timestamp}.md",
                        "text/markdown",
                        use_container_width=True,
                        key="download_markdown"
                    )

        with col2:
            st.metric("Messages", len(st.session_state.messages))
    else:
        st.info("Start a conversation to enable export")

# ============================================================================
# MAIN CHAT AREA
# ============================================================================

# st.markdown("### 💬 Chat")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            st.caption(f"Provider: {message['metadata'].get('provider')} | Model: {message['metadata'].get('model')}")
        if "response_time" in message:
            st.markdown(f"<div class='response-time'>⏱️ {message['response_time']:.2f}s</div>", unsafe_allow_html=True)

# Chat input
st.divider()

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_area(
        "Your message:",
        height=100,
        placeholder="Ask anything...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<div style='height: 4rem'></div>", unsafe_allow_html=True)
    send_button = st.button("✉️ Send", use_container_width=True)

# Handle message
if send_button and user_input:
    if not st.session_state.current_model or not st.session_state.current_provider:
        st.error("⚠️ Please select a provider and model first")
    else:
        # Include file content if a file was selected for use
        message_content = user_input

        if hasattr(st.session_state, 'current_file') and st.session_state.current_file:
            file_name = st.session_state.current_file
            file_content = st.session_state.file_contents.get(file_name, "")

            # Add file context to the message
            if file_content:
                message_content = f"File: {file_name}\n\nContent:\n{file_content[:2000]}\n\nQuestion: {user_input}"
                st.success(f"📎 Including file '{file_name}' in your query")
                st.session_state.current_file = None  # Reset after use

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": message_content,
            "file_attached": hasattr(st.session_state, 'current_file') and st.session_state.current_file is not None
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
            if hasattr(st.session_state, 'current_file') and st.session_state.current_file:
                st.caption(f"📎 File attached: {st.session_state.current_file}")

        # Get AI response
        with st.chat_message("assistant"):
            try:
                placeholder = st.empty()
                start_time = time.time()
                full_response = ""

                # Stream response
                for chunk in llm_manager.chat(
                    provider=st.session_state.current_provider,
                    model=st.session_state.current_model,
                    messages=[{"role": "user", "content": user_input}],
                    temperature=st.session_state.settings["temperature"],
                    top_p=st.session_state.settings["top_p"],
                    top_k=st.session_state.settings["top_k"],
                    num_predict=st.session_state.settings["num_predict"]
                ):
                    full_response += chunk
                    placeholder.markdown(full_response + " ▌")

                placeholder.markdown(full_response)
                response_time = time.time() - start_time

                # Add to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "response_time": response_time,
                    "metadata": {
                        "provider": st.session_state.current_provider,
                        "model": st.session_state.current_model
                    }
                })

                # Show metadata
                st.caption(f"⏱️ {response_time:.2f}s | {st.session_state.current_provider.upper()} | {st.session_state.current_model}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<center>
<p><b>Unified LLM Chat v1.0</b></p>

<p>© 2026 copyright all right reserved | Designed by ADITI GULATI</p>
<p style='font-size: 0.8rem; color: gray;'>Support for Ollama • OpenAI • Anthropic Claude</p>
</center>
""", unsafe_allow_html=True)