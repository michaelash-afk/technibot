# Technibot.py
"""
Technibot — Advanced single-file Streamlit app
- Chat: HuggingFace (flan-t5) + offline heuristic fallback
- Image Gen: HuggingFace Stable Diffusion inference + DeepAI text2img fallback
- Video Gen: DeepAI text2video (free-tier possible) + placeholder fallbacks
- TTS: gTTS (local, free) + espeak fallback message
- Games: 15 browser-playable adventurous/action games embedded via iframes/search pages
- UI: Light/Dark toggle, gradient theme accents, finalQuery wrapper toggle
- Keys: use Streamlit secrets or sidebar ephemeral inputs
- Files: generated media saved to /tmp and offered for download
"""

import os
import io
import json
import time
import textwrap
import tempfile
import base64
from typing import Optional, Dict, Any

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from gtts import gTTS

# -------------------------
# Page config & basic CSS
# -------------------------
st.set_page_config(page_title="Technibot", layout="wide", page_icon="🤖")
# default session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "finalQuery_enabled" not in st.session_state:
    st.session_state.finalQuery_enabled = True
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history

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
    @media(max-width:600px){{ .neon-title {{ font-size:20px; }} }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


apply_css()

# -------------------------
# Helper: secrets / env
# -------------------------
def get_secret(key: str) -> Optional[str]:
    # prefer st.secrets, else environment
    try:
        v = st.secrets.get(key)
        if v:
            return v
    except Exception:
        pass
    return os.getenv(key, None)

# Default keys (read from secrets or env). They are optional.
HUGGINGFACE_API_KEY = get_secret("HUGGINGFACE_API_KEY")
DEEPAI_API_KEY = get_secret("DEEPAI_API_KEY")
# Note: Some providers listed by user require paid access (OpenAI, Runway, etc.)
# We include placeholders and recommend adding keys to Streamlit secrets.

# -------------------------
# FinalQuery builder (for Chat)
# -------------------------
def build_final_query(user_input: str, prompt_extra: str) -> str:
    final = ("As an Advanced Pro Paid AI CHATBOT THAT GIVES SUGGESTIONS FOR TECH-RELATED PROBLEMS, "
             "GIVE SOLUTION TO " + user_input + ", "
             "GIVE SOLUTION TO " + prompt_extra + " TOOLS NEEDED TO HELP, VIDEOS TO WATCH, SITES TO VISIT, APPS NEEDED, STEPS TO TAKE, ETC.")
    return final

# -------------------------
# Offline heuristic fallback (if no keys)
# -------------------------
def heuristic_response(prompt: str) -> str:
    # A short helpful fallback that acts like a lightweight knowledge base
    p = prompt.lower()
    if "install" in p or "setup" in p or "deploy" in p:
        return ("OFFLINE GUIDE (heuristic):\n1. Identify OS & versions.\n2. Update packages & dependencies.\n3. Install required runtime (python/node).\n4. Reproduce minimal example and consult provider docs.\n5. If issues persist, capture logs and search error text.")
    if "speed" in p or "slow" in p:
        return ("OFFLINE GUIDE (heuristic):\n1. Check CPU/memory usage.\n2. Inspect startup services.\n3. Disable unnecessary autostart apps.\n4. Run disk cleanup and update drivers.")
    return ("OFFLINE HEURISTIC:\nNo online model key found. Add a Hugging Face or DeepAI token in Streamlit secrets to enable online AI. Meanwhile, try to be specific about OS, versions, and error messages for better advice.")

# -------------------------
# HuggingFace text generation (flan-t5-large)
# -------------------------
def hf_text_generation(prompt: str, model: str = "google/flan-t5-large", max_new_tokens: int = 256) -> str:
    token = HUGGINGFACE_API_KEY or os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        return "[HuggingFace key not found] " + heuristic_response(prompt)
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2}}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return f"[Hugging Face error {r.status_code}] {r.text}"
        data = r.json()
        # Many HF inference endpoints return list with "generated_text"
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        # Sometimes returns plain text or dict - provide reasonable fallback
        return json.dumps(data) if isinstance(data, (dict, list)) else str(data)
    except Exception as e:
        return f"[HuggingFace request failed] {e}"

# -------------------------
# HuggingFace image generation (Stable Diffusion) via inference
# -------------------------
def hf_image_generation(prompt: str, model: str = "stabilityai/stable-diffusion-2-1"):
    token = HUGGINGFACE_API_KEY or os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        return {"error": "HuggingFace key not found. Add HUGGINGFACE_API_KEY to Streamlit secrets."}
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            return {"error": f"Hugging Face image error {r.status_code}", "text": r.text}
        # Some HF models return image bytes; some return JSON with base64
        content_type = r.headers.get("content-type", "")
        if "image" in content_type:
            return {"image_bytes": r.content}
        # If JSON - try to extract image data
        data = r.json()
        # For safety, return JSON; caller will handle if contains url or base64
        return data
    except Exception as e:
        return {"error": f"HF image generation failed: {e}"}

# -------------------------
# DeepAI text2img / text2video wrappers (free tier possible)
# -------------------------
def deepai_text2img(prompt: str):
    key = DEEPAI_API_KEY or os.getenv("DEEPAI_API_KEY")
    if not key:
        return {"error": "DEEPAI_API_KEY not found in secrets. Add it for DeepAI calls."}
    try:
        r = requests.post("https://api.deepai.org/api/text2img", data={"text": prompt}, headers={"api-key": key}, timeout=60)
        if r.status_code != 200:
            return {"error": f"DeepAI text2img error {r.status_code}", "text": r.text}
        data = r.json()
        return data  # contains "output_url" commonly
    except Exception as e:
        return {"error": f"DeepAI request failed: {e}"}

def deepai_text2video(prompt: str):
    key = DEEPAI_API_KEY or os.getenv("DEEPAI_API_KEY")
    if not key:
        return {"error": "DEEPAI_API_KEY not found in secrets. Add it for DeepAI calls."}
    try:
        r = requests.post("https://api.deepai.org/api/text2video", data={"text": prompt}, headers={"api-key": key}, timeout=120)
        if r.status_code != 200:
            return {"error": f"DeepAI text2video error {r.status_code}", "text": r.text}
        data = r.json()
        return data
    except Exception as e:
        return {"error": f"DeepAI video request failed: {e}"}

# -------------------------
# Text-to-Speech using gTTS (free)
# -------------------------
def tts_gtts(text: str, lang: str = "en"):
    # returns path to mp3 file or dict {error:...}
    try:
        tts = gTTS(text=text, lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.write_to_fp(tmp)
        tmp.flush()
        tmp.close()
        return {"mp3_path": tmp.name}
    except Exception as e:
        return {"error": f"gTTS failed: {e}. Consider checking network access (gTTS uses Google translate TTS endpoint)."}

# -------------------------
# Utility: download file helper (for in-memory bytes)
# -------------------------
def download_bytes_as_button(data_bytes: bytes, file_name: str, mime: str):
    st.download_button(label=f"Download {file_name}", data=data_bytes, file_name=file_name, mime=mime)

# -------------------------
# Games list: 15 adventurous/action 2D games (embedding via search pages / playable embeds when allowed)
# -------------------------
GAMES = [
    {"title": "Super Mario World (fan)", "search": "https://www.kongregate.com/games?search=super%20mario"},
    {"title": "Mega Man 2 (fan)", "search": "https://www.kongregate.com/games?search=mega%20man"},
    {"title": "Castlevania-inspired", "search": "https://armorgames.com/search?q=castlevania"},
    {"title": "Hollow Knight-style", "search": "https://www.newgrounds.com/search?q=hollow%20knight"},
    {"title": "EarthBound-inspired RPG", "search": "https://www.kongregate.com/games?search=earthbound"},
    {"title": "Shovel Knight-style", "search": "https://www.miniclip.com/games/search/?q=shovel%20knight"},
    {"title": "Chrono Trigger fan-made", "search": "https://www.newgrounds.com/search?q=chrono%20trigger"},
    {"title": "The Messenger-inspired", "search": "https://www.kongregate.com/games?search=the%20messenger"},
    {"title": "Undertale-style fan", "search": "https://www.armorgames.com/search?q=undertale"},
    {"title": "Retro Action Platformer", "search": "https://www.miniclip.com/games/search/?q=platformer"},
    {"title": "Pixel Action Adventure", "search": "https://www.addictinggames.com/search?q=pixel%20platformer"},
    {"title": "Time-Travel RPG fan", "search": "https://www.kongregate.com/games?search=time%20travel%20rpg"},
    {"title": "Indie Metroidvania", "search": "https://www.newgrounds.com/search?q=metroidvania"},
    {"title": "Action Roguelike", "search": "https://www.addictinggames.com/search?q=roguelike"},
    {"title": "Ninja Platformer", "search": "https://www.kongregate.com/games?search=ninja%20platformer"},
]

# -------------------------
# UI: Sidebar
# -------------------------
def sidebar():
    st.sidebar.markdown("<div class='card'><div style='font-weight:700;font-size:20px'>Technibot</div><div class='muted'>AI Media, Chat & Games</div></div>", unsafe_allow_html=True)
    st.sidebar.markdown("---", unsafe_allow_html=True)
    # Theme toggle
    if st.sidebar.button("Toggle Light / Dark"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        apply_css()
        st.experimental_rerun()
    st.sidebar.markdown("### API Keys (optional)")
    # ephemeral inputs (session only)
    st.sidebar.text_input("HuggingFace API Key (hf_...)", key="hf_key_input", type="password", placeholder="hf_xxx")
    st.sidebar.text_input("DeepAI API Key (deepai-...)", key="deepai_key_input", type="password", placeholder="deepai-xxx")
    if st.sidebar.button("Apply keys to this session"):
        # apply to env for current process
        if st.session_state.hf_key_input:
            os.environ["HUGGINGFACE_API_KEY"] = st.session_state.hf_key_input
            st.success("HuggingFace key added to session env.")
        if st.session_state.deepai_key_input:
            os.environ["DEEPAI_API_KEY"] = st.session_state.deepai_key_input
            st.success("DeepAI key added to session env.")
        st.experimental_rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### FinalQuery (Chat)")
    st.sidebar.checkbox("Enable finalQuery wrapper", value=st.session_state.finalQuery_enabled, key="finalQuery_enabled")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Links & Notes")
    st.sidebar.markdown("- Add keys to Streamlit secrets for persistent access.")
    st.sidebar.markdown("- gTTS (free) is used for TTS by default.")
    st.sidebar.markdown("- Hugging Face and DeepAI have free tiers but require tokens.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Download sample secrets.toml")
    if st.sidebar.button("Download sample secrets.toml"):
        content = textwrap.dedent("""\
            # sample secrets.toml for Streamlit Cloud
            HUGGINGFACE_API_KEY = \"hf_...\"  # Hugging Face token (free tier available)
            DEEPAI_API_KEY = \"deepai-...\"  # DeepAI token (free tier available)
            # Optional: add other provider keys you want to use
        """)
        st.sidebar.download_button("Download secrets.toml", data=content, file_name="secrets.toml", mime="text/plain")

# -------------------------
# Header
# -------------------------
def top_header():
    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center;'><div><span class='neon-title'>Technibot</span><div class='muted'>AI-powered chat • media • games</div></div><div><small class='muted'>Theme: {}</small></div></div>".format(st.session_state.theme), unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)

# -------------------------
# Chat panel
# -------------------------
def chat_panel():
    st.markdown("<div class='card'><h3>Chat — Tech Help</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        user_input = st.text_area("Ask Technibot a tech question:", height=140)
        extra_prompt = st.text_input("Optional extra context (appended to finalQuery):")
        ai_choice = st.selectbox("Answer source", options=["HuggingFace (flan-t5-large)", "Knowledge Base (offline heuristic)"])
        use_final = st.checkbox("Wrap with finalQuery (include tools, videos, steps)", value=st.session_state.finalQuery_enabled)
        if st.button("Send"):
            if not user_input.strip():
                st.warning("Please type a question.")
            else:
                prompt_text = user_input.strip()
                if use_final:
                    prompt_text = build_final_query(user_input.strip(), extra_prompt.strip() or "N/A")
                if ai_choice.startswith("HuggingFace"):
                    ans = hf_text_generation(prompt_text)
                else:
                    ans = heuristic_response(prompt_text)
                st.session_state.messages.append({"role": "user", "text": user_input})
                st.session_state.messages.append({"role": "assistant", "text": ans})
                st.experimental_rerun()
    with col2:
        st.markdown("<div class='card'><h4>Chat History</h4></div>", unsafe_allow_html=True)
        for m in reversed(st.session_state.messages[-12:]):
            if m["role"] == "user":
                st.markdown(f"**You:** {m['text']}")
            else:
                st.markdown(f"**Technibot:** {m['text'][:500]}{'...' if len(m['text'])>500 else ''}")

# -------------------------
# Image panel
# -------------------------
def image_panel():
    st.markdown("<div class='card'><h3>Image Generation</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        prompt = st.text_area("Describe the image you want (style, lighting, composition):", value="Futuristic neon city at night, ultra-detailed, cinematic")
        model_choice = st.selectbox("Image Model", options=["HuggingFace Stable Diffusion (inference)", "DeepAI text2img (fallback)"])
        if st.button("Generate Image"):
            if model_choice.startswith("HuggingFace"):
                res = hf_image_generation(prompt)
                if isinstance(res, dict) and "error" in res:
                    st.error(res["error"])
                elif isinstance(res, dict) and "image_bytes" in res:
                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmpf.write(res["image_bytes"])
                    tmpf.close()
                    st.image(tmpf.name, use_column_width=True)
                    with open(tmpf.name, "rb") as f:
                        st.download_button("Download Image", f, file_name="technibot_image.png", mime="image/png")
                else:
                    # show whatever HF returned (json or base64)
                    st.json(res)
            else:
                res = deepai_text2img(prompt)
                if "error" in res:
                    st.error(res["error"])
                else:
                    # DeepAI typically provides an output_url
                    url = res.get("output_url") or res.get("output_urls") or res.get("output")
                    if isinstance(url, list):
                        url = url[0]
                    if url:
                        st.image(url, use_column_width=True)
                        # download the image bytes
                        try:
                            r = requests.get(url, timeout=30)
                            if r.status_code == 200:
                                download_bytes_as_button(r.content, "technibot_image.png", "image/png")
                        except Exception as e:
                            st.warning(f"Could not fetch image bytes for download: {e}")
                    else:
                        st.json(res)
    with col2:
        st.markdown("<div class='card'><h4>Image Tips</h4><div class='muted'>Use style tags like 'cinematic', 'ultra-detailed', 'photorealistic', '8k' to improve results.</div></div>", unsafe_allow_html=True)

# -------------------------
# Video & TTS panel
# -------------------------
def video_panel():
    st.markdown("<div class='card'><h3>Video Generation & Voiceover</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        v_prompt = st.text_area("Describe the short video you'd like (scene, style, duration seconds):", value="Coding timelapse, fast cuts, neon overlays, 12 seconds")
        v_provider = st.selectbox("Video provider", options=["DeepAI text2video (free tier)", "Placeholder (external editors: Kapwing/Veed/Clipchamp)"])
        if st.button("Generate Video"):
            if v_provider.startswith("DeepAI"):
                res = deepai_text2video(v_prompt)
                if "error" in res:
                    st.error(res["error"])
                else:
                    # DeepAI often returns output_url in JSON
                    url = res.get("output_url") or res.get("id") or res.get("output")
                    if isinstance(url, list):
                        url = url[0]
                    if url and isinstance(url, str) and url.startswith("http"):
                        st.video(url)
                        # attempt to let user download the video bytes
                        try:
                            r = requests.get(url, stream=True, timeout=60)
                            if r.status_code == 200:
                                # read bytes
                                vid_bytes = r.content
                                download_bytes_as_button(vid_bytes, "technibot_video.mp4", "video/mp4")
                        except Exception as e:
                            st.warning(f"Could not fetch video bytes for download: {e}")
                    else:
                        st.json(res)
            else:
                st.info("External editors (Kapwing/Veed/Clipchamp) require redirect/embedding. Use their web UIs directly.")
    with col2:
        st.markdown("<div class='card'><h4>Generate Voiceover (TTS)</h4></div>", unsafe_allow_html=True)
        tts_text = st.text_area("Text for narration (TTS):", height=120)
        tts_choice = st.selectbox("TTS Provider (free-first)", options=["gTTS (Google Translate TTS, free)", "espeak NG (local placeholder)"])
        if st.button("Generate Voiceover"):
            if not tts_text.strip():
                st.warning("Enter text to convert to speech.")
            else:
                if tts_choice.startswith("gTTS"):
                    res = tts_gtts(tts_text)
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        mp3_path = res["mp3_path"]
                        st.audio(mp3_path)
                        with open(mp3_path, "rb") as f:
                            st.download_button("Download MP3", f, file_name="technibot_tts.mp3", mime="audio/mpeg")
                else:
                    st.info("espeak NG is a local CLI TTS; to use it deploy the app on a server with espeak-ng installed and call it via subprocess.")

# -------------------------
# Games panel
# -------------------------
def games_panel():
    st.markdown("<div class='card'><h3>Games — Adventure & Action (browser)</h3></div>", unsafe_allow_html=True)
    st.markdown("<div class='game-grid'>", unsafe_allow_html=True)
    for g in GAMES:
        title = g["title"]
        search_url = g["search"]
        with st.container():
            st.markdown(f"<div class='game-card'><b>{title}</b><div class='muted'>Platform search / playable options</div></div>", unsafe_allow_html=True)
            cols = st.columns([3,1])
            with cols[0]:
                # iframe preview (sites may block embedding; handle gracefully)
                try:
                    components.iframe(search_url, height=240)
                except Exception:
                    st.markdown(f"[Open {title} on platform]({search_url})")
            with cols[1]:
                st.markdown(f"[Open]({search_url})")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Settings & API key guide (toml example)
# -------------------------
API_KEYS_DOC = """
### How to get API keys & add to Streamlit secrets (toml)
1. Hugging Face (text & image models)
   - Sign up at https://huggingface.co/
   - Go to Settings → Access Tokens → New token (select 'read' scope).
   - Add to Streamlit secrets:
     ```
     HUGGINGFACE_API_KEY = "hf_..."
     ```
2. DeepAI (text2img / text2video)
   - Sign up at https://deepai.org/
   - Copy your API key and add:
     ```
     DEEPAI_API_KEY = "deepai-..."
     ```
3. (Optional) Other providers: add as environment variables or in secrets.toml.
"""

def settings_panel():
    st.markdown("<div class='card'><h3>Settings & API Key Guide</h3></div>", unsafe_allow_html=True)
    st.markdown(API_KEYS_DOC)
    st.markdown("Example `secrets.toml` for Streamlit (paste into Streamlit secrets):")
    st.code(textwrap.dedent("""\
        HUGGINGFACE_API_KEY = "hf_..."
        DEEPAI_API_KEY = "deepai-..."
    """), language="toml")

# -------------------------
# Main
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
    st.markdown("<div class='muted'>Technibot — demo scaffold. Add provider API keys in Streamlit secrets to enable full AI features (Hugging Face, DeepAI). All free-tier usage where supported.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

