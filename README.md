# 🤖 Unified LLM Chat Application

**A production-ready Streamlit app supporting Ollama (local), OpenAI (GPT-4), and Anthropic (Claude)**

---

## ⚡ Quick Start 

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Configure `.env`
```env
# LOCAL: Ollama (Free, runs on your computer)
OLLAMA_BASE_URL=http://localhost:11434

# CLOUD: OpenAI (GPT-4, GPT-3.5)
OPENAI_API_KEY=sk-your-key-here

# CLOUD: Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### **3. Run App**
```bash
python -m streamlit run app_main.py
```
Opens at: `http://localhost:8501`

---

## 🏠 Ollama: Run LLMs Locally (No API Needed)

### **What is Ollama?**
Open-source tool to run LLMs on your computer. Download models, chat locally. Free. Private.

### **Installation**

**Windows/Mac:**
- Download: https://ollama.ai
- Install and run

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### **Start Ollama Server** (Keep this running)
```bash
ollama serve
# Runs at http://localhost:11434
```

### **Download & Chat with Models** (In another terminal)
```bash
# Download a model (one-time)
ollama pull mistral          # Fast, high quality (5GB)
ollama pull neural-chat      # Chat optimized (5GB)
ollama pull llama2           # General purpose (4GB)
ollama pull tinyllama        # Lightweight (1.1GB)

# Chat directly in terminal
ollama run mistral
```

### **Available Ollama Models**

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| tinyllama | 1.1B | ⭐⭐⭐⭐⭐ | ⭐⭐ | Testing, fast |
| orca-mini | 3B | ⭐⭐⭐⭐ | ⭐⭐⭐ | Quick responses |
| neural-chat | 7B | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Chat, conversation |
| mistral | 7B | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Balanced, coding |
| llama2 | 7B | ⭐⭐⭐ | ⭐⭐⭐⭐ | General purpose |
| zephyr | 7B | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High quality |
| solar | 10.7B | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Complex reasoning |

---

## 🚀 This App: Multi-Provider Chat

### **What Does It Do?**

Beautiful web chat interface where you pick a provider:
- **Ollama:** Use local models (free, private)
- **OpenAI:** Use GPT-4/GPT-3.5 (paid, cloud)
- **Anthropic:** Use Claude (paid, cloud)

All via one app. Switch providers instantly. Same settings work everywhere.

### **How It Works**

```
Browser (You type message)
    ↓
Streamlit UI (app_main.py)
    ↓
Provider Selection (Which LLM?)
    ├─ Ollama → Local models via HTTP
    ├─ OpenAI → GPT-4/3.5 via API
    └─ Anthropic → Claude via API
    ↓
Stream Response (Word by word)
    ↓
Browser (You see response)
```

### **File Overview**

| File | Purpose |
|------|---------|
| `app_main.py` | Streamlit web interface (chat UI) |
| `llm_provider.py` | Backend logic (connects to providers) |
| `requirements.txt` | Python packages needed |
| `.env` | Your API keys & settings |

---

## 🎮 Features

### **1. Provider Selection**
Dropdown in sidebar. Choose:
- 🏠 **Ollama** (local models, free)
- 🔴 **OpenAI** (if API key set)
- 🟣 **Anthropic** (if API key set)


### **2. Model Selection**
Auto-loads models for selected provider:
- **Ollama:** Shows installed models (mistral, llama2, etc.)
- **OpenAI:** GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
- **Anthropic:** Claude 3 Sonnet, Haiku, Opus


### **3. Advanced Settings** (Sidebar Sliders)

| Setting | Range | Effect | Test |
|---------|-------|--------|------|
| **Temperature** | 0.0-2.0 | Creativity | Low=precise facts, High=creative |
| **Top-P** | 0.0-1.0 | Diversity | Controls response variety |
| **Top-K** | 1-100 | Token limit | How many options model considers |
| **Max Response** | 128-4096 | Length limit | Short answers vs long essays |


### **4. Real-time Streaming**
Response appears word-by-word as it's generated (not all at once)


### **5. Chat History**
All messages saved. Shows:
- Who said what (user/assistant)
- Which provider/model was used
- Response time
- Metadata

### **6. Export Chat**
Download entire conversation as JSON (includes messages, settings, metadata)

---

## ⚙️ Configuration Guide

### **`.env` File - Complete Reference**

```env
# Which provider to use initially
PROVIDER=ollama

# === OLLAMA (Local) ===
OLLAMA_BASE_URL=http://localhost:11434
# For remote: OLLAMA_BASE_URL=https://your-server.com:11434

# === OPENAI (Cloud) ===
OPENAI_API_KEY=sk-...your-key-from-platform.openai.com...
OPENAI_MODEL=gpt-4-turbo

# === ANTHROPIC (Cloud) ===
ANTHROPIC_API_KEY=sk-ant-...your-key-from-console.anthropic.com...
CLAUDE_MODEL=claude-3-sonnet-20240229

# === Default Chat Settings ===
DEFAULT_TEMPERATURE=0.7        # 0=precise, 2=creative
DEFAULT_TOP_P=0.9
DEFAULT_TOP_K=40
DEFAULT_NUM_PREDICT=512        # Max response tokens
DEFAULT_NUM_CONTEXT=2048       # Memory window
```

### **Getting API Keys**

**OpenAI:**
1. Go to https://platform.openai.com/api-keys
2. Sign up/login
3. Create API key
4. Add to `.env`: `OPENAI_API_KEY=sk-...`

**Anthropic:**
1. Go to https://console.anthropic.com
2. Sign up/login
3. Create API key
4. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

---

## 🎓 Understanding Settings

### **Temperature (Creativity Control)**

```
Temperature = 0.0
↓
Response: "The capital of France is Paris."
(Always same, factual, precise)

Temperature = 0.7 (Default)
↓
Response: "Paris is France's capital..."
(Balanced, natural, slightly varied)

Temperature = 1.5
↓
Response: "France's heart beats in Paris... 
City of light and wonder..."
(Creative, varied, poetic)
```

**Use Case:**
- **0.0-0.3:** Coding, math, facts (you want consistency)
- **0.7:** General chat, emails (balanced)
- **1.2-2.0:** Creative writing, brainstorming (you want variety)

### **Max Response Length**

```
Max = 256 tokens (default)
→ Short, focused answers
→ Fast responses

Max = 2048 tokens
→ Long explanations
→ Detailed code examples
→ Slower but more complete
```

### **Context Window (Memory)**

```
2048 tokens (default)
→ Remembers last ~2000 words of conversation

4096 tokens
→ Better memory, slower processing

512 tokens
→ Fast but forgets more easily
```

---

## 📊 Provider Comparison

| Feature | Ollama | OpenAI (GPT-4) | Anthropic (Claude) |
|---------|--------|---|---|
| **Cost** | Free | $0.03/1K out tokens | $0.015/1K out tokens |
| **Setup** | Download & run | Get API key | Get API key |
| **Speed** | Depends on hardware | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Requires** | Nothing (local) | Internet + payment | Internet + payment |
| **Best For** | Development, testing | Production, complex | Production, coding |

---

## 🏃 Scenarios

### **Scenario 1: Local-Only (No API Keys)**
```bash
# Terminal 1
ollama serve

# Terminal 2
streamlit run app_main.py
# Select: Ollama → mistral → Chat locally
```

### **Scenario 2: Compare Ollama vs GPT-4**
```bash
# Terminal 1
ollama serve

# .env has both:
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=sk-...

# App browser:
# Ask question with Ollama (mistral)
# Switch provider to OpenAI
# Ask same question with GPT-4
# Compare responses side-by-side (in chat history)
```

### **Scenario 3: Show Different Temperatures**
```bash
# Ask: "What is AI?"

# Temperature = 0.1 → Precise technical definition
# Temperature = 0.7 → Balanced explanation
# Temperature = 1.5 → Creative, philosophical explanation

# Watch how same prompt produces different responses
```

### **Scenario 4: Speed Test**
```bash
# Use tinyllama (1.1GB) → Very fast local
# Use mistral (5GB) → Better quality, slower
# Use GPT-4 → Highest quality, cloud speed
```

---

## 🚀 Deployment

### **Option 1: Streamlit Cloud** (Easiest - 5 minutes)

**Step 1: Push to GitHub**
```bash
git init
git add .
git commit -m "LLM Chat App"
git remote add origin https://github.com/aditi-gulati/llm-chat.git
git push -u origin main
```

**Step 2: Deploy**
- Go to https://streamlit.io/cloud
- Select your repo
- Select `app_main.py`
- Click Deploy

**Step 3: Add Secrets**
- In Streamlit dashboard → Settings → Secrets
- Add: `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`
- (Ollama won't work here - requires local server)

**Result:** App live at `https://aditi-llm.streamlit.app/`

### **Option 2: Docker** (For any server)

```bash
# Create Dockerfile in project root:
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app_main.py", "--server.address=0.0.0.0"]

# Build & run:
docker build -t llm-chat .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-... \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  llm-chat
```

### **Option 3: Self-Hosted (Your server)**
```bash
# On your server:
git clone https://github.com/aditi-gulati/llm-chat.git
cd llm-chat
pip install -r requirements.txt

# Run:
streamlit run app_main.py
```

---

## 🆘 Quick Troubleshooting

### **"Cannot connect to Ollama"**
```bash
# Check if running:
curl http://localhost:11434/api/tags

# If error, start it:
ollama serve
```

### **"No providers available"**
- Ensure `.env` file exists
- Add at least one: OLLAMA_BASE_URL or OPENAI_API_KEY
- Restart app

### **"API key invalid"**
- Verify key in `.env`
- Check key hasn't expired
- For OpenAI: https://platform.openai.com/api-keys

### **"Out of memory"**
- Use smaller Ollama model: `ollama pull tinyllama`
- Or use cloud (OpenAI/Anthropic)

### **"Streamlit port in use"**
```bash
streamlit run app_main.py --server.port 8502
```

---

## 📚 Code Structure

### **`app_main.py` - Streamlit UI**
- Page setup & layout
- Sidebar: Provider/Model selection, Settings
- Main area: Chat display, Message input
- Import: `llm_provider.get_llm_manager()`

### **`llm_provider.py` - Backend Logic**
- `OllamaProvider` class → HTTP to local/remote Ollama
- `OpenAIProvider` class → OpenAI API calls
- `AnthropicProvider` class → Anthropic API calls
- `UnifiedLLMManager` → Manages all providers

### **How They Work Together**
```python
# app_main.py
manager = get_llm_manager()  # Gets manager from llm_provider.py

# User selects provider (Ollama/OpenAI/Anthropic)
for chunk in manager.chat(
    provider="openai",
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
):
    # Stream response word by word
    print(chunk, end="")
```

---

### **Token Economics**

```
OpenAI GPT-4 Turbo:
- Input: $0.01 per 1K tokens
- Output: $0.03 per 1K tokens
- ~250 tokens per page

Claude 3 Sonnet:
- Input: $0.003 per 1K tokens
- Output: $0.015 per 1K tokens
- Most cost-effective
```

### **Model Capabilities**

| Task | Ollama 7B | GPT-4 | Claude 3 |
|------|-----------|-------|---------|
| General Chat | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Code Gen | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Reasoning | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Long Context | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### **Extending the App**

**Add Function Calling:**
```python
# Enable GPT-4 tools/Claude's function calling
# In llm_provider.py, add tools parameter
```

**Add RAG (Retrieval Augmented Generation):**
```python
# Add vector database (FAISS, Pinecone)
# Embed docs → Store vectors
# Query → Retrieve + Feed to LLM
```

**Add Caching:**
```python
# Cache responses for cost savings
# Use prompt-response caching APIs
```

---

## 📝 Quick Reference

### **Start Local (No API Keys)**
```bash
# Terminal 1
ollama serve

# Terminal 2
streamlit run app_main.py
# → Select Ollama, pick model, chat
```

### **Use Cloud APIs**
```bash
# Update .env with:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

streamlit run app_main.py
# → Select OpenAI/Anthropic, chat
```

### **Deploy**
```bash
# GitHub → Streamlit Cloud 
git push origin main
# → https://streamlit.io/cloud
```

---

## 🎓 Learn More

- **Ollama:** https://ollama.ai
- **OpenAI API:** https://platform.openai.com/docs
- **OpenAI API Key:** https://platform.openai.com/settings/organization/api-keys
- **Anthropic:** https://docs.anthropic.com
- **Anthropic Key:** https://platform.claude.com/settings/keys
- **Streamlit:** https://docs.streamlit.io

