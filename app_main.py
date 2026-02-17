"""
�� Ollama Chat Application - Cloud-Based with Advanced Features
- Uses Ollama via Python API (no local installation needed)
- Cloud/Remote Ollama support
- Model selection in UI
- Memory/Context management
- Advanced settings (temperature, top-p, top-k)
- Real-time streaming responses
- Chat history & export
- Streamlit web interface
"""

import streamlit as st
import requests
import json
from datetime import datetime
import time
from typing import List, Dict, Generator
import os

# Import Ollama Models Management
from ollama_models import OllamaModels, DEFAULT_MODELS, MODEL_RECOMMENDATIONS

# ============================================================================
# CONFIGURATION
# ============================================================================

# IMPORTANT: Set your Ollama server URL
# For cloud: https://your-cloud-server.com:11434
# For local: http://localhost:11434
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Page config
st.set_page_config(
    page_title="�� Ollama Chat Pro",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stChatMessage { border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0; }
    .response-time { font-size: 0.8rem; color: gray; margin-top: 0.5rem; }
    .model-info { padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; margin: 0.5rem 0; }
    .settings-box { padding: 1rem; background-color: #fff3e0; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_model" not in st.session_state:
    st.session_state.current_model = None

if "models_list" not in st.session_state:
    st.session_state.models_list = []

if "settings" not in st.session_state:
    st.session_state.settings = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "num_predict": 256,
        "context_length": 2048,
        "system_prompt": "You are a helpful, intelligent, and friendly AI assistant."
    }

# ============================================================================
# OLLAMA CLIENT - PYTHON-BASED
# ============================================================================

class OllamaClientPython:
    """
    Python-based Ollama client
    Works with cloud/remote Ollama servers
    No local installation needed - just Python HTTP requests
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip('/')

    def get_models(self) -> List[str]:
        """Get list of available models from Ollama server"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            return sorted(models)
        except Exception as e:
            st.error(f"❌ Cannot connect to Ollama: {str(e)}")
            return []

    def check_server(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: int = 256,
        system: str = "You are a helpful assistant."
    ) -> str:
        """Send message and get response"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": num_predict,
                "system": system,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300
            )

            response.raise_for_status()
            data = response.json()
            return data.get('message', {}).get('content', '')

        except Exception as e:
            raise Exception(f"Chat error: {str(e)}")

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: int = 256,
        system: str = "You are a helpful assistant."
    ) -> Generator[str, None, None]:
        """Stream chat response"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": num_predict,
                "system": system,
                "stream": True
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=300
            )

            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        yield data.get('message', {}).get('content', '')
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            raise Exception(f"Streaming error: {str(e)}")


# Initialize client
client = OllamaClientPython(OLLAMA_BASE_URL)

# Initialize Models Manager
models_manager = OllamaModels(OLLAMA_BASE_URL)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("# �� Ollama Chat Pro - Advanced Edition")
st.markdown("*Python-Based Cloud/Remote LLM Chat with Advanced Settings*")
st.markdown(f"**Server:** {OLLAMA_BASE_URL}")
st.divider()

# ============================================================================
# SIDEBAR - SETTINGS & CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings & Configuration")
    st.divider()

    # Check server connection
    if not client.check_server():
        st.error(f"""
        ❌ **Cannot Connect to Ollama**

        Server: {OLLAMA_BASE_URL}

        **Solutions:**
        1. Ensure Ollama is running
        2. Check server URL is correct
        3. Check network connectivity

        For local: http://localhost:11434
        For cloud: https://your-cloud-server.com:11434
        """)

    # Model Selection
    st.markdown("### �� Model Selection")

    if st.button("�� Refresh Models", use_container_width=True):
        st.session_state.models_list = []

    if not st.session_state.models_list:
        with st.spinner("Loading models..."):
            models = models_manager.get_installed_models()
            st.session_state.models_list = models
    else:
        models = st.session_state.models_list

    if models:
        selected_model = st.selectbox(
            "Available Models:",
            models,
            key="model_select"
        )
        st.session_state.current_model = selected_model

        # Show model info
        model_info = models_manager.get_model_info(selected_model)
        with st.expander("�� Model Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Size:** {model_info.get('size', 'Unknown')}")
                st.write(f"**Speed:** {model_info.get('speed', 'Unknown')}")
            with col2:
                st.write(f"**Quality:** {model_info.get('quality', 'Unknown')}")
                st.write(f"**Description:** {model_info.get('description', '')}")

        st.success(f"✅ Selected: {selected_model}")
    else:
        st.warning("No models available. Check Ollama connection.")

        # Show recommended models to pull
        st.markdown("#### �� Recommended Models to Pull:")
        for model in DEFAULT_MODELS:
            col1, col2 = st.columns([3, 1])
            with col1:
                model_info = models_manager.get_model_info(model)
                st.write(f"**{model}** - {model_info.get('size')}")
            with col2:
                if st.button(f"Pull", key=f"pull_{model}", use_container_width=True):
                    with st.spinner(f"Pulling {model}..."):
                        if models_manager.pull_model(model):
                            st.success(f"✅ {model} pulled!")
                            st.rerun()
                        else:
                            st.error(f"Failed to pull {model}")

    st.divider()

    # Advanced Settings
    st.markdown("### ��️ Advanced Model Settings")

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
            "Top-P (Nucleus)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.settings["top_p"],
            step=0.05,
            help="0.9=More consistent, 1.0=More random"
        )
        st.session_state.settings["top_p"] = top_p

    with col2:
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=100,
            value=st.session_state.settings["top_k"],
            step=1,
            help="Limit responses to K most likely tokens"
        )
        st.session_state.settings["top_k"] = top_k

        num_predict = st.slider(
            "Max Response Length",
            min_value=128,
            max_value=4096,
            value=st.session_state.settings["num_predict"],
            step=128,
            help="Maximum tokens in response"
        )
        st.session_state.settings["num_predict"] = num_predict

    st.divider()

    # Context & Memory
    st.markdown("### �� Memory & Context")

    context_length = st.slider(
        "Context Window Size",
        min_value=512,
        max_value=4096,
        value=st.session_state.settings["context_length"],
        step=256,
        help="How many previous tokens to consider"
    )
    st.session_state.settings["context_length"] = context_length

    st.info(f"""
    **Context Info:**
    - Current messages: {len(st.session_state.messages)}
    - Context window: {context_length} tokens
    - Max response: {num_predict} tokens
    """)

    st.divider()

    # System Prompt
    st.markdown("### �� System Prompt")

    system_prompt = st.text_area(
        "Define AI Behavior:",
        value=st.session_state.settings["system_prompt"],
        height=100,
        help="Customize how the AI responds"
    )
    st.session_state.settings["system_prompt"] = system_prompt

    if st.button("�� Reset Prompt", use_container_width=True):
        st.session_state.settings["system_prompt"] = "You are a helpful, intelligent, and friendly AI assistant."
        st.rerun()

    st.divider()

    # Chat Management
    st.markdown("### �� Chat Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("��️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.success("✅ Chat cleared!")
            st.rerun()

    with col2:
        if st.button("�� Statistics", use_container_width=True):
            st.info(f"""
            **Chat Statistics:**
            - Total messages: {len(st.session_state.messages)}
            - User messages: {len([m for m in st.session_state.messages if m['role'] == 'user'])}
            - Assistant messages: {len([m for m in st.session_state.messages if m['role'] == 'assistant'])}
            """)

    st.divider()

    # Export Chat
    st.markdown("### �� Export Chat")

    if st.session_state.messages:
        export_data = {
            "model": st.session_state.current_model,
            "timestamp": datetime.now().isoformat(),
            "settings": st.session_state.settings,
            "messages": st.session_state.messages
        }

        export_json = json.dumps(export_data, indent=2)
        st.download_button(
            "�� Download Chat (JSON)",
            export_json,
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )
    else:
        st.caption("�� Send messages to enable export")

# ============================================================================
# MAIN CHAT AREA
# ============================================================================

st.markdown("### �� Chat Window")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "response_time" in message:
            st.markdown(f"<div class='response-time'>⏱️ {message['response_time']:.2f}s</div>", unsafe_allow_html=True)

# Chat input
st.divider()

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_area(
        "Your message:",
        height=100,
        placeholder="Ask anything... (supports multi-line)",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<div style='height: 4rem'></div>", unsafe_allow_html=True)
    send_button = st.button("�� Send", use_container_width=True)

# Handle message sending
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    if st.session_state.current_model:
        with st.chat_message("assistant"):
            try:
                placeholder = st.empty()
                start_time = time.time()
                full_response = ""

                # Stream response
                for chunk in client.chat_stream(
                    model=st.session_state.current_model,
                    messages=[{"role": "user", "content": user_input}],
                    temperature=st.session_state.settings["temperature"],
                    top_p=st.session_state.settings["top_p"],
                    top_k=st.session_state.settings["top_k"],
                    num_predict=st.session_state.settings["num_predict"],
                    system=st.session_state.settings["system_prompt"]
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
                    "model": st.session_state.current_model
                })

                # Show response time
                st.markdown(f"<div class='response-time'>⏱️ {response_time:.2f}s | Model: {st.session_state.current_model}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please select a model first")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<center>
<p>�� <b>Ollama Chat Pro v2.0</b> | Python-Based Cloud Ollama</p>
<p style='font-size: 0.8rem; color: gray;'>
Advanced LLM Chat with Model Selection • Memory Management • Real-time Streaming
</p>
</center>
""", unsafe_allow_html=True)
