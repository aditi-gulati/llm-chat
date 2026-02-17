"""
�� Unified LLM Chat Application
Support for Ollama, OpenAI (GPT-4, GPT-3.5), and Anthropic (Claude Sonnet, Haiku)
Real-time streaming, advanced settings, chat history & export
"""

import streamlit as st
import json
from datetime import datetime
import time
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import unified LLM manager
from llm_provider import get_llm_manager

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="�� Unified LLM Chat",
    page_icon="��",
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

# ============================================================================
# INITIALIZE LLM MANAGER
# ============================================================================

llm_manager = get_llm_manager()
available_providers = llm_manager.get_available_providers()

if not available_providers:
    st.error("❌ No LLM providers configured. Please set API keys in .env file")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.markdown("# �� Unified LLM Chat")
st.markdown("*Support for Ollama, OpenAI (GPT-4, GPT-3.5), and Claude (Sonnet, Haiku)*")
st.divider()

# ============================================================================
# SIDEBAR - PROVIDER & MODEL SELECTION
# ============================================================================

with st.sidebar:
    st.markdown("## �� Provider & Model Selection")
    st.divider()

    # Provider Selection
    st.markdown("### �� Select Provider")

    provider_options = {
        "ollama": "�� Ollama (Local/Remote)",
        "openai": "�� OpenAI (GPT-4, GPT-3.5)",
        "anthropic": "�� Anthropic (Claude)"
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
    st.markdown("### �� Select Model")

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
            with st.expander("�� Model Details"):
                for key, value in model_info.items():
                    if key != "name":
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")

        st.success(f"✅ Selected: {selected_model}")
    else:
        st.warning(f"No models available for {selected_provider}")

    st.divider()

    # Advanced Settings
    st.markdown("### ��️ Advanced Settings")

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

    # Chat Management
    st.markdown("### �� Chat Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("��️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("�� Stats", use_container_width=True):
            st.info(f"Messages: {len(st.session_state.messages)}")

    st.divider()

    # Export
    st.markdown("### �� Export Chat")

    if st.session_state.messages:
        export_data = {
            "provider": st.session_state.current_provider,
            "model": st.session_state.current_model,
            "timestamp": datetime.now().isoformat(),
            "settings": st.session_state.settings,
            "messages": st.session_state.messages
        }

        export_json = json.dumps(export_data, indent=2)
        st.download_button(
            "�� Download Chat",
            export_json,
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )

# ============================================================================
# MAIN CHAT AREA
# ============================================================================

st.markdown("### �� Chat")

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
    send_button = st.button("�� Send", use_container_width=True)

# Handle message
if send_button and user_input:
    if not st.session_state.current_model or not st.session_state.current_provider:
        st.error("⚠️ Please select a provider and model first")
    else:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

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
<p>�� <b>Unified LLM Chat v1.0</b></p>
<p style='font-size: 0.8rem; color: gray;'>Support for Ollama • OpenAI • Anthropic Claude</p>
</center>
""", unsafe_allow_html=True)
