# technibot_app.py
"""
Technibot — Advanced single-file Streamlit app
Features:
 - Chat: choose AI provider (OpenAI / HuggingFace / Knowledge Base fallback)
 - Image Gen: HuggingFace Stable Diffusion wrapper + fallback
 - Video Gen: wrappers for DeepAI/Runway placeholders (requires keys)
 - Voice (TTS): multiple provider toggles (Google TTS placeholder, TTS-MP3, espeak)
 - Games: 15 HTML/JS browser games embedded via iframes/components (Kongregate, ArmorGames, Miniclip, AddictingGames, Newgrounds)
 - UI: Light/Dark toggle, gradient theme options
 - finalQuery toggle & button for Chat
 - Secrets support (st.secrets or .env)
"""

import os
import json
import time
import tempfile
import base64
import textwrap
from typing import Optional, Dict, Any, List

import requests
import streamlit as st
import streamlit.components.v1 as components

# -------------------------
# Page config & defaults
# -------------------------
st.set_page_config(page_title="Technibot", layout="wide", initial_sidebar_state="expanded", page_icon="🤖")

# -------------------------
# Load API keys from streamlit secrets (preferred) or env (local dev)
# -------------------------
def get_secret(key: str) -> Optional[str]:
    try:
        return st.secrets.get(key)  # Streamlit secrets (preferred)
    except Exception:
        return os.getenv(key)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = get_secret("HUGGINGFACE_API_KEY")
PEXELS_API_KEY = get_secret("PEXELS_API_KEY")
DEEPAI_API_KEY = get_secret("DEEPAI_API_KEY")
RUNWAY_API_KEY = get_secret("RUNWAY_API_KEY")
TTSMP3_API_KEY = get_secret("TTSMP3_API_KEY")

# -------------------------
# App session defaults
# -------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"   # 'dark' or 'light'

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history: list of dicts {"role":"user"/"assistant", "text":...}

if "finalQuery_enabled" not in st.session_state:
    st.session_state.finalQuery_enabled = True

# -------------------------
# Styling (light/dark, neon)
# -------------------------
def apply_css():
    theme = st.session_state.theme
    bg = "#071018" if theme == "dark" else "#f8fafc"
    panel = "#0b1620" if theme == "dark" else "#ffffff"
    text = "#e6f6f0" if theme == "dark" else "#091226"
    muted = "#8da0b3" if theme == "dark" else "#6b7280"
    css = f"""
    <style>
    .stApp {{ background: radial-gradient(circle at 10% 10%, rgba(122,0,255,0.06), transparent 8%), radial-gradient(circle at 90% 90%, rgba(0,255,213,0.03), transparent 8%), {bg}; color: {text}; }}
    .card {{ background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.03); }}
    .neon-title {{ font-weight:700; font-size:28px; background: -webkit-linear-gradient(#fff, #8ef9e3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 18px rgba(0,255,213,0.08); }}
    .muted {{ color: {muted}; font-size:13px; }}
    .small-btn {{ background: linear-gradient(90deg,#00ffd5,#7a00ff); border-radius:8px; padding:8px 12px; color:#001; font-weight:700; }}
    .game-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap: 12px; }}
    .game-card {{ padding:8px; border-radius:10px; background: rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.02); }}
    @media(max-width:600px){ .neon-title { font-size:20px; } }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_css()

# -------------------------
# Utility helpers
# -------------------------
def download_json(data: dict, filename: str = "data.json"):
    b64 = base64.b64encode(json.dumps(data, indent=2).encode()).decode()
    href = f"data:application/json;base64,{b64}"
    st.markdown(f"[Download {filename}]({href})", unsafe_allow_html=True)

def heuristic_response(prompt: str) -> str:
    # Useful offline fallback
    if "install" in prompt.lower():
        return ("OFFLINE FALLBACK:\n1) Detect OS\n2) Update packages\n3) Install required packages\n4) Follow vendor docs\n\n(Use a real model by providing API keys to get full responses.)")
    return ("OFFLINE FALLBACK:\nI couldn't find an API key. Try adding OPENAI_API_KEY or HUGGINGFACE_API_KEY to Streamlit secrets.\n\nQuick tips:\n- Reproduce the problem\n- Gather logs\n- Search error messages\n- Try minimal example")

# -------------------------
# AI Wrappers
# -------------------------
def openai_chat(prompt: str, max_tokens: int = 600) -> str:
    if not OPENAI_API_KEY:
        return "[OpenAI API key not found] " + heuristic_response(prompt)
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": "You are Technibot, an expert technology assistant."},
                     {"role": "user", "content": prompt}],
        "temperature": 0.15,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=body, timeout=30)
    if r.status_code != 200:
        return f"[OpenAI error {r.status_code}] {r.text}"
    try:
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(r.text)

def hf_text_generation(prompt: str, model: str = "google/flan-t5-large", max_new_tokens: int = 512) -> str:
    # HuggingFace text-inference API
    if not HUGGINGFACE_API_KEY:
        return "[HuggingFace API key not found] " + heuristic_response(prompt)
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    body = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2}}
    r = requests.post(url, headers=headers, json=body, timeout=40)
    if r.status_code != 200:
        return f"[HuggingFace error {r.status_code}] {r.text}"
    try:
        data = r.json()
        # often returns list with generated_text
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return json.dumps(data)
    except Exception:
        return str(r.text)

def build_final_query(user_input: str, prompt_extra: str) -> str:
    finalQuery = ("As an Advanced Pro Paid AI CHATBOT THAT GIVES SUGGESTIONS FOR TECH-RELATED PROBLEMS, "
                  "GIVE SOLUTION TO " + user_input + ", "
                  "GIVE SOLUTION TO " + prompt_extra + " TOOLS NEEDED TO HELP, VIDEOS TO WATCH, SITES TO VISIT, APPS NEEDED, STEPS TO TAKE, ETC.")
    return finalQuery

# Image generation: HF stable-diffusion (inference)
def hf_image_generation(prompt: str, model: str = "stabilityai/stable-diffusion-2"):
    if not HUGGINGFACE_API_KEY:
        return {"error": "HuggingFace key not found. Add HUGGINGFACE_API_KEY to secrets."}
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    body = {"inputs": prompt}
    r = requests.post(url, headers=headers, json=body, timeout=60)
    if r.status_code != 200:
        return {"error": f"HF error {r.status_code}", "text": r.text}
    # Some models return image bytes, others return JSON - handle both
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        return r.json()
    else:
        return {"image_bytes": r.content}

# Video generation placeholder: DeepAI / Runway (these require separate onboarding & keys)
def deepai_video_generate(prompt: str):
    if not DEEPAI_API_KEY:
        return {"error": "DEEPAI_API_KEY missing in secrets."}
    url = "https://api.deepai.org/api/text2video"  # example endpoint
    headers = {"api-key": DEEPAI_API_KEY}
    r = requests.post(url, headers=headers, data={"text": prompt}, timeout=60)
    if r.status_code != 200:
        return {"error": f"DeepAI error {r.status_code}", "text": r.text}
    return r.json()

# TTS example: TTSMP3.com (simple) or local espeak fallback
def tts_via_ttsmp3(text: str, voice: str = "Joey"):
    if not TTSMP3_API_KEY:
        # fallback: return a data URL using gTTS-like approach not available offline
        return {"error": "TTSMP3 API key missing; use espeak on local machine for TTS fallback."}
    # Example wrapper — the real TTSMP3 REST API will differ; check provider docs
    url = "https://api.ttsmp3.com/v1/text_to_speech"
    headers = {"Authorization": f"Bearer {TTSMP3_API_KEY}", "Content-Type": "application/json"}
    body = {"text": text, "voice": voice}
    r = requests.post(url, headers=headers, json=body, timeout=30)
    if r.status_code != 200:
        return {"error": f"TTSMP3 error {r.status_code}", "text": r.text}
    return r.json()

# -------------------------
# Game Data: 15 adventurous/action 2D games (browser-friendly embeddings)
# Each entry contains: title, platforms_to_open (search urls), embed_url (if available)
# We'll embed via platform search results or direct embedable game page when possible.
# -------------------------
GAMES = [
    {"title": "Super Mario World (fan)", "search": "https://www.kongregate.com/games?search=super%20mario", "platforms": ["kongregate", "newgrounds"]},
    {"title": "Mega Man 2 (fan)", "search": "https://www.kongregate.com/games?search=mega%20man", "platforms": ["kongregate", "armorgames"]},
    {"title": "Castlevania-inspired", "search": "https://armorgames.com/search?q=castlevania", "platforms": ["armorgames"]},
    {"title": "Hollow Knight-style", "search": "https://www.newgrounds.com/search?q=hollow%20knight", "platforms": ["newgrounds"]},
    {"title": "EarthBound-inspired RPG fan", "search": "https://www.kongregate.com/games?search=earthbound", "platforms": ["kongregate"]},
    {"title": "Shovel Knight-style", "search": "https://www.miniclip.com/games/search/?q=shovel%20knight", "platforms": ["miniclip"]},
    {"title": "Chrono Trigger fan-made", "search": "https://www.newgrounds.com/search?q=chrono%20trigger", "platforms": ["newgrounds"]},
    {"title": "The Messenger-inspired", "search": "https://www.kongregate.com/games?search=the%20messenger", "platforms": ["kongregate"]},
    {"title": "Undertale-style fan", "search": "https://www.armorgames.com/search?q=undertale", "platforms": ["armorgames"]},
    {"title": "Retro Action Platformer", "search": "https://www.miniclip.com/games/search/?q=platformer", "platforms": ["miniclip"]},
    {"title": "Pixel Action Adventure", "search": "https://www.addictinggames.com/search?q=pixel%20platformer", "platforms": ["addictinggames"]},
    {"title": "Time-Travel RPG fan", "search": "https://www.kongregate.com/games?search=time%20travel%20rpg", "platforms": ["kongregate"]},
    {"title": "Indie Metroidvania", "search": "https://www.newgrounds.com/search?q=metroidvania", "platforms": ["newgrounds"]},
    {"title": "Action Roguelike", "search": "https://www.addictinggames.com/search?q=roguelike", "platforms": ["addictinggames"]},
    {"title": "Ninja Platformer", "search": "https://www.kongregate.com/games?search=ninja%20platformer", "platforms": ["kongregate", "miniclip"]},
]

# -------------------------
# UI Components
# -------------------------
def sidebar():
    st.sidebar.markdown("<div class='card'><div class='neon-title'>Technibot</div><div class='muted'>AI Media, Chat & Games</div></div>", unsafe_allow_html=True)
    st.sidebar.markdown("---", unsafe_allow_html=True)
    # Theme toggle
    if st.sidebar.button("Toggle Light/Dark"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        apply_css()
        st.experimental_rerun()
    st.sidebar.markdown("### Keys (optional)")
    st.sidebar.text_input("OpenAI API Key", key="openai_key_input", type="password", placeholder="sk-...")
    st.sidebar.text_input("HuggingFace API Key", key="hf_key_input", type="password", placeholder="hf_...")
    st.sidebar.text_input("DeepAI API Key", key="deepai_key_input", type="password", placeholder="deepai-...")
    st.sidebar.text_input("Runway API Key", key="runway_key_input", type="password", placeholder="runway-...")
    st.sidebar.text_input("Pexels API Key", key="pexels_key_input", type="password", placeholder="pexels-...")
    if st.sidebar.button("Apply keys (session only)"):
        if st.session_state.openai_key_input:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key_input
            st.success("OpenAI key added to session env.")
        if st.session_state.hf_key_input:
            os.environ["HUGGINGFACE_API_KEY"] = st.session_state.hf_key_input
            st.success("HuggingFace key added to session env.")
        if st.session_state.deepai_key_input:
            os.environ["DEEPAI_API_KEY"] = st.session_state.deepai_key_input
            st.success("DeepAI key added.")
        if st.session_state.runway_key_input:
            os.environ["RUNWAY_API_KEY"] = st.session_state.runway_key_input
            st.success("Runway key added.")
        if st.session_state.pexels_key_input:
            os.environ["PEXELS_API_KEY"] = st.session_state.pexels_key_input
            st.success("Pexels key added.")
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Final Query (chat)")
    st.sidebar.checkbox("Enable finalQuery wrapper (chat)", value=st.session_state.finalQuery_enabled, key="finalQuery_enabled")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Links")
    st.sidebar.markdown("- [How to get API keys & add to Streamlit secrets](#)")
    st.sidebar.markdown("- Download tech_knowledge.json below")
    if st.sidebar.button("Download sample tech_knowledge.json"):
        sample = {"meta": {"project":"Technibot", "version":"1.0"}, "topics":["Web dev","AI","Games"]}
        download_json(sample, filename="tech_knowledge.json")

def top_header():
    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center;'><div><span class='neon-title'>Technibot</span><div class='muted'>AI-powered chat • media • games</div></div><div><small class='muted'>Theme: {}</small></div></div>".format(st.session_state.theme), unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)

# -------------------------
# Chat Panel
# -------------------------
def chat_panel():
    st.markdown("<div class='card'><h3>Chat — Tech Help</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        user_input = st.text_area("Ask Technibot a tech question (e.g., 'How to speed up Windows boot?')", height=120)
        extra_prompt = st.text_input("Optional extra context (will be appended to finalQuery)")
        ai_choice = st.selectbox("Answer source", options=["OpenAI (GPT-3.5)","HuggingFace (Flan-T5)","Knowledge Base (offline heuristic)"])
        use_final = st.checkbox("Wrap with finalQuery (include tools, videos, steps)", value=st.session_state.finalQuery_enabled)
        if st.button("Send"):
            if not user_input.strip():
                st.warning("Please type a question.")
            else:
                # Build prompt based on toggles
                prompt_text = user_input.strip()
                if use_final:
                    prompt_text = build_final_query(user_input.strip(), extra_prompt.strip() or "N/A")
                # choose provider
                if ai_choice.startswith("OpenAI"):
                    ans = openai_chat(prompt_text)
                elif ai_choice.startswith("HuggingFace"):
                    ans = hf_text_generation(prompt_text)
                else:
                    ans = heuristic_response(prompt_text)
                st.session_state.messages.append({"role":"user","text":user_input})
                st.session_state.messages.append({"role":"assistant","text":ans})
                st.experimental_rerun()
    with col2:
        st.markdown("<div class='card'><h4>Chat History</h4></div>", unsafe_allow_html=True)
        for m in st.session_state.messages[::-1][:8]:
            if m["role"] == "user":
                st.markdown(f"**You:** {m['text']}")
            else:
                st.markdown(f"**Technibot:** {m['text'][:400]}{'...' if len(m['text'])>400 else ''}")

# -------------------------
# Image generation panel
# -------------------------
def image_panel():
    st.markdown("<div class='card'><h3>Image Generation</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        prompt = st.text_area("Describe the image you want (style, lighting, composition):", value="Futuristic neon city at night, ultra-detailed, cinematic")
        model_choice = st.selectbox("Image Model", options=["Stable Diffusion (HuggingFace)","Pixray (external)"])
        if st.button("Generate Image"):
            if model_choice.startswith("Stable"):
                res = hf_image_generation(prompt)
                if "error" in res:
                    st.error(res["error"])
                elif "image_bytes" in res:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmp.write(res["image_bytes"])
                    tmp.close()
                    st.image(tmp.name, use_column_width=True)
                    with open(tmp.name,"rb") as f:
                        st.download_button("Download image", f, file_name="technibot_image.png")
                else:
                    st.json(res)
            else:
                st.info("Pixray is an external interactive tool; embed or call its API if you have credentials.")
    with col2:
        st.markdown("<div class='card'><h4>Image Tips</h4><div class='muted'>Use style tags like 'cinematic', 'ultra-detailed', 'photorealistic', '8k' to improve results.</div></div>", unsafe_allow_html=True)

# -------------------------
# Video generation panel
# -------------------------
def video_panel():
    st.markdown("<div class='card'><h3>Video Generation</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        v_prompt = st.text_area("Describe the video you'd like (short scene or concept):", value="Coding timelapse, fast cuts, neon overlays, 20 seconds")
        v_model = st.selectbox("Video generation provider", options=["DeepAI (text2video)","Runway ML (placeholder)","Veed/Kapwing (external editors)"])
        if st.button("Generate Video"):
            if v_model.startswith("DeepAI"):
                res = deepai_video_generate(v_prompt)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.json(res)
            elif v_model.startswith("Runway"):
                st.info("Runway ML requires their SDK/API & model selection — add RUNWAY_API_KEY in secrets to enable.")
            else:
                st.info("Veed/Kapwing are editor platforms; they often require a redirect or embedding their editor UI.")
    with col2:
        st.markdown("<div class='card'><h4>Video Voiceover (TTS)</h4></div>", unsafe_allow_html=True)
        tts_text = st.text_area("Text-to-speech text (for narration)", height=80)
        tts_provider = st.selectbox("TTS provider", options=["Google TTS (external)","TTSMP3 (API)","espeak NG (local)"])
        if st.button("Generate TTS"):
            if tts_provider.startswith("TTSMP3"):
                res = tts_via_ttsmp3(tts_text)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.json(res)
            else:
                st.info("Use Google TTS or local espeak for TTS output; add provider keys to use cloud TTS.")

# -------------------------
# Games Panel: embed HTML games or platform search pages
# -------------------------
def games_panel():
    st.markdown("<div class='card'><h3>Games — Adventure & Action (browser)</h3></div>", unsafe_allow_html=True)
    st.markdown("<div class='game-grid'>", unsafe_allow_html=True)
    for g in GAMES:
        title = g["title"]
        search_url = g["search"]
        # show quick card with iframe to search results (responsive)
        with st.container():
            st.markdown(f"<div class='game-card'><b>{title}</b><div class='muted'>Platform search / playable options</div></div>", unsafe_allow_html=True)
            # Provide buttons to open search in new tab or embed small preview
            cols = st.columns([3,1])
            with cols[0]:
                # Small embedded iframe preview (if allowed by site)
                try:
                    components.iframe(search_url, height=240)
                except Exception:
                    st.markdown(f"[Open {title} on platform]({search_url})")
            with cols[1]:
                st.write("")
                st.markdown(f"[Open]({search_url})")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Settings & API Key guide
# -------------------------
API_KEYS_DOC = """
### How to get API keys & add to Streamlit secrets (toml)
1. **OpenAI (Chat & GPT-3.5):**
   - Sign up at https://platform.openai.com/
   - Create an API key in the Dashboard → API keys.
   - In Streamlit Cloud: App → Manage App → Secrets, add:
     ```
     OPENAI_API_KEY = "sk-..."
     ```
2. **Hugging Face (Text & Image models):**
   - Sign up: https://huggingface.co/
   - Go to Settings → Access Tokens → New token.
   - Add to Streamlit secrets:
     ```
     HUGGINGFACE_API_KEY = "hf_..."
     ```
3. **DeepAI (Video generation):**
   - https://deepai.org/
   - Get API key and add:
     ```
     DEEPAI_API_KEY = "deepai-..."
     ```
4. **Runway / other paid APIs:**
   - Follow provider docs, place keys in secrets as: RUNWAY_API_KEY = "..."
5. **TTS providers (e.g., TTSMP3):**
   - Add TTSMP3_API_KEY = "..."

To add secrets: Go to Streamlit Cloud -> Select app -> Settings -> Secrets -> paste the key lines and Save.
"""

def settings_panel():
    st.markdown("<div class='card'><h3>Settings & API Key Guide</h3></div>", unsafe_allow_html=True)
    st.markdown(API_KEYS_DOC)
    st.markdown("Example `secrets.toml` for Streamlit (paste into Streamlit secrets):")
    st.code(textwrap.dedent("""\
        OPENAI_API_KEY = "sk-..."
        HUGGINGFACE_API_KEY = "hf_..."
        DEEPAI_API_KEY = "deepai-..."
        RUNWAY_API_KEY = "runway-..."
        PEXELS_API_KEY = "pexels-..."
        TTSMP3_API_KEY = "ttsmp3-..."
    """), language="toml")

# -------------------------
# Main layout
# -------------------------
def main():
    sidebar()
    top_header()

    tabs = st.tabs(["Chat", "Image", "Video & TTS", "Games", "Settings"])
    with tabs[0]:
        chat_panel()
    with tabs[1]:
        image_panel()
    with tabs[2]:
        video_panel()
    with tabs[3]:
        games_panel()
    with tabs[4]:
        settings_panel()

    st.markdown("---")
    st.markdown("<div class='muted'>Technibot — demo scaffold. Add provider API keys in Streamlit secrets to enable full non-redirected AI features.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
