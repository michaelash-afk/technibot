# app.py
"""
Technibot - All-in-one Streamlit app
Features:
- Chatbot for tech help (finalQuery enforced)
- Image & video generation interfaces (Hugging Face / Pexels)
- Embedded mini-games (Snake, simple platformer)
- Futuristic UI with light/dark toggle
- Local caching + downloadable manifest/service-worker/tech_knowledge.json
- Single-file deployment for Streamlit (GitHub -> Streamlit sharing)
"""

import os
import json
import time
import base64
import tempfile
import textwrap
from typing import List, Dict, Any, Optional

import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------
# CONFIG & HELPERS
# ---------------------------

st.set_page_config(
    page_title="Technibot — Advanced Tech Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Environment keys (optional)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")

# Tech knowledge JSON (embedded)
TECH_KNOWLEDGE = {
    "categories": [
        {"id": "os", "title": "Operating Systems", "items": ["Windows", "Linux", "macOS"]},
        {"id": "web", "title": "Web Dev", "items": ["HTML", "CSS", "JavaScript", "Streamlit"]},
        {"id": "ai", "title": "AI & ML", "items": ["HuggingFace", "OpenAI", "Stable Diffusion"]},
    ],
    "meta": {"project": "Technibot", "version": "1.0", "author": "Technibot Generator"}
}

# Downloadable assets (manifest & service worker)
MANIFEST_JSON = {
    "name": "Technibot",
    "short_name": "Technibot",
    "start_url": "/",
    "display": "standalone",
    "icons": [{"src": "icon.png", "sizes": "512x512", "type": "image/png"}],
    "theme_color": "#0ff",
    "background_color": "#000"
}

SERVICE_WORKER_JS = """
// Minimalistic service worker stub for caching static assets - Streamlit can't register this client-side normally,
// but it's included for progressive web app export and for your static hosting if you extract files.
self.addEventListener('install', function(event) {
  event.waitUntil(caches.open('technibot-v1').then(function(cache) {
    return cache.addAll(['/','/index.html']);
  }));
});
self.addEventListener('fetch', function(event) {
  event.respondWith(caches.match(event.request).then(function(resp) {
    return resp || fetch(event.request);
  }));
});
"""

# Utility: download button wrapper for JSON
def download_json_button(data, filename="data.json", label="Download"):
    b64 = base64.b64encode(json.dumps(data, indent=2).encode()).decode()
    href = f"data:application/json;base64,{b64}"
    st.markdown(f"[{label}]({href})", unsafe_allow_html=True)

# Caching decorator for generated content
@st.cache_data(show_spinner=False)
def cached_api_post(url: str, headers: Dict[str, str], payload: Any = None, json_body: Any = None, timeout: int = 30):
    try:
        r = requests.post(url, headers=headers, data=payload, json=json_body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "status_code": getattr(e, "response", None)}

# ---------------------------
# UI THEME (light/dark)
# ---------------------------

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# CSS theme + neon accent
def local_css():
    theme = st.session_state.theme
    # simple neon gradient backgrounds and styles
    css = f"""
    <style>
    :root {{
      --bg-color: {'#0b0f14' if theme=='dark' else '#f8fafc'};
      --panel-color: {'#071018' if theme=='dark' else '#ffffff'};
      --muted: {'#8da0b3' if theme=='dark' else '#6b7280'};
      --accent: linear-gradient(90deg,#00ffd5,#7a00ff);
      --neon: rgba(0,255,213,0.08);
      --glass: rgba(255,255,255,0.02);
    }}
    .stApp {{
      background: radial-gradient(ellipse at 10% 10%, rgba(122,0,255,0.06), transparent 10%),
                  radial-gradient(ellipse at 90% 90%, rgba(0,255,213,0.03), transparent 10%),
                  var(--bg-color);
      color: {'#dbeafe' if theme=='dark' else '#0f172a'};
    }}
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(2,6,23,0.6);
      border: 1px solid rgba(255,255,255,0.03);
    }}
    .neon-title {{
      font-family: 'Segoe UI', Roboto, sans-serif;
      font-weight: 700;
      font-size: 28px;
      background: -webkit-linear-gradient(#fff, #8ef9e3);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-shadow: 0 0 18px rgba(0,255,213,0.12);
    }}
    .muted {{ color: var(--muted); font-size: 12px }}
    .small-btn {{
      background: linear-gradient(90deg,#00ffd5,#7a00ff);
      border: none;
      padding: 8px 12px;
      color: #001;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
    }}
    /* center HTML game canvas */
    .game-container {{
      display:flex; justify-content:center; align-items:center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply CSS
local_css()

# ---------------------------
# FINAL QUERY LOGIC (your required finalQuery)
# ---------------------------

def build_final_query(user_input: str, prompt: str) -> str:
    # Use your exact wording
    finalQuery = (
        "As an Advanced Pro Paid AI CHATBOT THAT GIVES SUGGESTIONS FOR TECH-RELATED PROBLEMS, "
        "GIVE SOLUTION TO " + user_input + ", "
        "GIVE SOLUTION TO " + prompt + " TOOLS NEEDED TO HELP, VIDEOS TO WATCH, SITES TO VISIT, APPS NEEDED, STEPS TO TAKE, ETC."
    )
    return finalQuery

# Execution engine: tries OpenAI -> HuggingFace -> local fallback
def run_text_generation(prompt: str, max_tokens: int = 800) -> str:
    # Priority: OpenAI if key is present, else Hugging Face Inference API, else local heuristic
    if OPENAI_API_KEY:
        try:
            # Use OpenAI ChatCompletions (classic). The user may replace this with their own implementation.
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {
                "model": "gpt-4o-mini" if False else "gpt-4o-mini",  # placeholder; user can edit
                "messages": [{"role": "system", "content": "You are Technibot, an expert helpful assistant."},
                             {"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.15
            }
            r = requests.post(url, headers=headers, json=body, timeout=30)
            r.raise_for_status()
            data = r.json()
            # adapt to response shapes
            content = ""
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
            return content or json.dumps(data)
        except Exception as e:
            return f"[OpenAI error] {str(e)}\n\nPrompt used:\n{prompt}"
    elif HUGGINGFACE_API_KEY:
        try:
            # Using the Hugging Face text-generation inference endpoint
            url = "https://api-inference.huggingface.co/models/gpt2"  # placeholder small model; you can choose better models
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"}
            body = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.2}}
            r = requests.post(url, headers=headers, json=body, timeout=40)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
            return json.dumps(data)
        except Exception as e:
            return f"[HuggingFace error] {str(e)}\n\nPrompt used:\n{prompt}"
    else:
        # Local heuristic fallback (not a real LLM): produce structured guidance based on keywords
        return heuristic_response(prompt)

def heuristic_response(prompt: str) -> str:
    # A deterministic but helpful fallback - good for demos and offline use.
    lines = []
    lines.append("TECHNIBOT OFFLINE-HEURISTIC: (no external API key found). Here's structured guidance based on your request.")
    if "install" in prompt.lower() or "setup" in prompt.lower():
        lines.append("\n1) Summary: This looks like an installation/setup request.")
        lines.append("2) Tools: installer, admin rights, internet, package manager (apt, brew, choco).")
        lines.append("3) Steps:\n   - Step 1: Identify OS.\n   - Step 2: Update system packages.\n   - Step 3: Install required tools.\n   - Step 4: Configure and verify.")
        lines.append("4) Videos: Search YouTube 'install <tool> on <os>'.")
        lines.append("5) Apps & Sites: official docs, StackOverflow, GitHub issues.")
    else:
        lines.append("\n1) Analysis: I parsed a tech question but no LLM key present.")
        lines.append("2) Quick tips: Break problem into (a) reproduce, (b) collect logs, (c) search exact error, (d) try minimal repro.")
        lines.append("3) Tools: terminal, network analyzer (wireshark), logs, browser devtools.")
        lines.append("4) Next steps: copy-paste error into StackOverflow and GitHub issues; try local troubleshooting steps.")
    lines.append("\nIf you provide an API key for OpenAI or HuggingFace the assistant will produce a long, tailored plan (tools, videos, sites, apps, step-by-step instructions).")
    return "\n".join(lines)

# ---------------------------
# IMAGE & VIDEO GENERATION (simple wrappers)
# ---------------------------

@st.cache_data(show_spinner=False)
def generate_image_with_hf(prompt: str, model: str = "stabilityai/stable-diffusion-2", size: str = "512x512"):
    if not HUGGINGFACE_API_KEY:
        return {"error": "HUGGINGFACE_API_KEY not set. Please set it to enable image generation."}
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        # The inference endpoint might return image bytes directly or JSON with artifacts
        if "application/json" in content_type:
            return resp.json()
        else:
            return {"image_bytes": resp.content}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def search_videos_pexels(query: str, per_page: int = 5):
    if not PEXELS_API_KEY:
        return {"error": "PEXELS_API_KEY not set."}
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": per_page}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# GAMES (embedded HTML/JS)
# ---------------------------

SNAKE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Snake - Technibot</title>
  <style>
    body { margin:0; background: linear-gradient(180deg,#001219,#001e2e); display:flex; align-items:center; justify-content:center; height:100vh; }
    canvas { background:#061826; border-radius:12px; box-shadow:0 10px 30px rgba(0,0,0,0.6); }
  </style>
</head>
<body>
  <canvas id="game" width="480" height="480"></canvas>
  <script>
  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');
  const grid = 20;
  let snake = [{x:9, y:9}];
  let dir = {x:0, y:0};
  let food = {x:5, y:5};
  let speed = 8;
  let last = 0;
  function loop(time) {
    if (!last || time - last > 1000 / speed) {
      update();
      draw();
      last = time;
    }
    requestAnimationFrame(loop);
  }
  function update() {
    const head = {x: snake[0].x + dir.x, y: snake[0].y + dir.y};
    if (head.x < 0) head.x = grid-1;
    if (head.y < 0) head.y = grid-1;
    if (head.x >= grid) head.x = 0;
    if (head.y >= grid) head.y = 0;
    snake.unshift(head);
    if (head.x === food.x && head.y === food.y) {
      food = {x: Math.floor(Math.random() * grid), y: Math.floor(Math.random() * grid)};
      speed = Math.min(22, speed + 0.5);
    } else {
      snake.pop();
    }
    // collision
    for (let i=1;i<snake.length;i++) {
      if (snake[i].x === head.x && snake[i].y === head.y) {
        snake = [{x:9,y:9}];
        dir = {x:0,y:0};
        food = {x:5,y:5};
        speed = 8;
      }
    }
  }
  function draw() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    // draw food
    ctx.fillStyle="#00ffd5";
    ctx.fillRect(food.x * (canvas.width/grid), food.y * (canvas.width/grid), canvas.width/grid-2, canvas.width/grid-2);
    // draw snake
    ctx.fillStyle="#7a00ff";
    for (let s of snake) {
      ctx.fillRect(s.x * (canvas.width/grid), s.y * (canvas.width/grid), canvas.width/grid-2, canvas.width/grid-2);
    }
  }
  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp') dir={x:0,y:-1};
    if (e.key === 'ArrowDown') dir={x:0,y:1};
    if (e.key === 'ArrowLeft') dir={x:-1,y:0};
    if (e.key === 'ArrowRight') dir={x:1,y:0};
  });
  requestAnimationFrame(loop);
  </script>
</body>
</html>
"""

PLATFORMER_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Mini Platformer</title>
<style>
  body { margin:0; background: linear-gradient(180deg,#020617,#091229); display:flex; align-items:center; justify-content:center; height:100vh; }
  #canvas { border-radius:12px; box-shadow: 0 12px 36px rgba(0,0,0,0.6); background: linear-gradient(#07102b,#081022); }
</style>
</head>
<body>
<canvas id="canvas" width="900" height="420"></canvas>
<script>
const canvas=document.getElementById('canvas'); const ctx=canvas.getContext('2d');
let player={x:50,y:300,w:28,h:40,vx:0,vy:0,jumping:false};
let keys={};
const gravity=0.8;
const platforms=[{x:0,y:360,w:900,h:60},{x:200,y:260,w:140,h:16},{x:420,y:200,w:140,h:16},{x:640,y:140,w:160,h:16}];
function loop(){
  // physics
  player.vy+=gravity;
  player.x+=player.vx; player.y+=player.vy;
  // collision with bounds
  if(player.y+player.h>360){ player.y=300; player.vy=0; player.jumping=false;}
  // platform collision
  platforms.forEach(p=>{
    if(player.x+player.w>p.x && player.x<p.x+p.w && player.y+player.h>p.y && player.y+player.h < p.y + 20){
      player.y=p.y-player.h; player.vy=0; player.jumping=false;
    }
  });
  // friction and input
  player.vx*=0.9;
  if(keys['ArrowLeft']) player.vx=-4;
  if(keys['ArrowRight']) player.vx=4;
  if(keys['Space'] && !player.jumping){ player.vy=-14; player.jumping=true; }
  draw();
  requestAnimationFrame(loop);
}
function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // background glow
  const g=ctx.createLinearGradient(0,0,0,canvas.height); g.addColorStop(0,'#07102b'); g.addColorStop(1,'#081022'); ctx.fillStyle=g; ctx.fillRect(0,0,canvas.width,canvas.height);
  // platforms
  ctx.fillStyle='#00ffd5'; platforms.forEach(p=>ctx.fillRect(p.x,p.y,p.w,p.h));
  // player (neon)
  ctx.fillStyle='#7a00ff'; ctx.fillRect(player.x,player.y,player.w,player.h);
}
window.addEventListener('keydown', (e)=> keys[e.key]=true);
window.addEventListener('keyup', (e)=> keys[e.key]=false);
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

# ---------------------------
# STREAMLIT LAYOUT
# ---------------------------

def sidebar_ui():
    st.sidebar.markdown("<div class='card'><div class='neon-title'>Technibot</div>"
                        "<div class='muted'>Advanced tech assistant • chat • media • games</div></div>",
                        unsafe_allow_html=True)
    st.sidebar.markdown("---", unsafe_allow_html=True)
    st.sidebar.button("Toggle light/dark", on_click=toggle_theme)
    st.sidebar.markdown("## Quick tools")
    if st.sidebar.button("Download manifest.json"):
        download_json_button(MANIFEST_JSON, filename="manifest.json", label="manifest.json")
    if st.sidebar.button("Download service-worker.js"):
        b64 = base64.b64encode(SERVICE_WORKER_JS.encode()).decode()
        href = f"data:text/javascript;base64,{b64}"
        st.sidebar.markdown(f"[Download service-worker.js]({href})", unsafe_allow_html=True)
    if st.sidebar.button("Download tech_knowledge.json"):
        download_json_button(TECH_KNOWLEDGE, filename="tech_knowledge.json", label="tech_knowledge.json")
    st.sidebar.markdown("---")
    st.sidebar.markdown("API Keys (optional):")
    st.sidebar.text_input("HuggingFace API Key", key="hf_key_input", value=HUGGINGFACE_API_KEY, type="password")
    st.sidebar.text_input("OpenAI API Key", key="openai_key_input", value=OPENAI_API_KEY, type="password")
    st.sidebar.text_input("Pexels API Key", key="pexels_key_input", value=PEXELS_API_KEY, type="password")
    if st.sidebar.button("Apply keys"):
        # Save temporarily into env (only for this session) so the app can use them
        if st.session_state.hf_key_input:
            os.environ["HUGGINGFACE_API_KEY"] = st.session_state.hf_key_input
        if st.session_state.openai_key_input:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key_input
        if st.session_state.pexels_key_input:
            os.environ["PEXELS_API_KEY"] = st.session_state.pexels_key_input
        st.experimental_rerun()

def main_ui():
    st.markdown(
        "<div class='card'><div style='display:flex;justify-content:space-between;align-items:center;'>"
        "<div><span class='neon-title'>Technibot</span><div class='muted'>Your pro tech support & media studio</div></div>"
        "<div><small class='muted'>Theme: {}</small></div></div></div>".format(st.session_state.theme),
        unsafe_allow_html=True
    )
    cols = st.columns([2, 1])
    with cols[0]:
        chat_panel()
    with cols[1]:
        tools_panel()

def chat_panel():
    st.markdown("<div class='card'><h3>Chat — Tech Help</h3></div>", unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Input
    with st.form("chat_form", clear_on_submit=False):
        user_input = st.text_area("Ask Technibot (e.g., 'How to fix slow Windows startup?')", height=100, key="user_input")
        additional_prompt = st.text_input("Optional extra prompt/context (this will be appended)", placeholder="Add more context to the query", key="extra_prompt")
        submitted = st.form_submit_button("Ask")
        if submitted:
            if not user_input.strip():
                st.warning("Please type your question.")
            else:
                # Build finalQuery per your exact requirement:
                final_q = build_final_query(user_input.strip(), additional_prompt.strip())
                # Append user message and show "thinking"
                st.session_state.messages.append({"role": "user", "text": user_input.strip()})
                with st.spinner("Generating response..."):
                    response = run_text_generation(final_q, max_tokens=800)
                st.session_state.messages.append({"role": "assistant", "text": response})
                st.experimental_rerun()

    # Display conversation
    for m in st.session_state.messages[::-1]:
        if m["role"] == "user":
            st.markdown(f"<div class='card' style='background:linear-gradient(90deg,#00121f,#071a2b);'><b>You:</b><div style='margin-top:6px'>{m['text']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='card' style='background:linear-gradient(90deg,#071018,#08102a);'><b>Technibot:</b><div style='margin-top:6px;white-space:pre-wrap'>{m['text']}</div></div>", unsafe_allow_html=True)

def tools_panel():
    st.markdown("<div class='card'><h4>Media Studio</h4></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Image Gen", "Video Search", "Games"])
    with tab1:
        image_generation_ui()
    with tab2:
        video_search_ui()
    with tab3:
        games_ui()

def image_generation_ui():
    st.markdown("### Image Generation (text to image)")
    prompt = st.text_input("Image prompt (describe the scene, style, lighting):", value="Futuristic city neon at night, cinematic, ultra-detailed")
    model = st.selectbox("Model (HuggingFace)", options=["stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-xl"], index=0)
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            res = generate_image_with_hf(prompt, model=model)
        if "error" in res:
            st.error(res["error"])
            if not HUGGINGFACE_API_KEY:
                st.info("You can get a HuggingFace API key (free tier) at https://huggingface.co/settings/tokens and paste it in the sidebar.")
        else:
            # If bytes returned:
            if "image_bytes" in res:
                b = res["image_bytes"]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(b)
                tmp.close()
                st.image(tmp.name, caption="Generated image (from HF model)", use_column_width=True)
                with open(tmp.name, "rb") as f:
                    st.download_button("Download image", f, file_name="technibot_image.png")
            else:
                st.json(res)

def video_search_ui():
    st.markdown("### Video search (Pexels fallback)")
    q = st.text_input("Search videos for (e.g., 'coding timelapse')", value="coding timelapse")
    if st.button("Search Videos"):
        with st.spinner("Searching videos..."):
            res = search_videos_pexels(q)
        if "error" in res:
            st.error(res["error"])
            if not PEXELS_API_KEY:
                st.info("Pexels offers free developer keys at https://www.pexels.com/api/")
        else:
            videos = res.get("videos", [])
            if not videos:
                st.info("No videos found.")
            for vid in videos:
                st.video(vid["video_files"][0]["link"])
                st.write(f"🎥 {vid.get('user', {}).get('name', 'Unknown')} — {vid.get('duration', 0)}s")

def games_ui():
    st.markdown("### Games — press arrow keys / space")
    st.markdown("**Snake**")
    components.html(SNAKE_HTML, height=520)
    st.markdown("**Mini Platformer**")
    components.html(PLATFORMER_HTML, height=520)

# ---------------------------
# Automatic finalQuery run at startup (only once)
# ---------------------------

def auto_run_final_query_at_startup():
    # We'll execute the finalQuery with a demo prompt automatically if session flag not set.
    if not st.session_state.get("auto_ran", False):
        demo_input = "help me troubleshoot a slow laptop startup"
        demo_extra = "Include tools, videos, sites, apps, and step by step commands for Windows 10"
        final_q = build_final_query(demo_input, demo_extra)
        st.session_state.auto_ran = True
        # Run in background-like manner (but immediately) and show the result in a hidden variable to be shown in UI
        try:
            res = run_text_generation(final_q, max_tokens=600)
        except Exception as e:
            res = f"[auto-run error] {e}"
        # Save to session as initial assistant message
        st.session_state.messages = st.session_state.get("messages", []) + [
            {"role": "user", "text": demo_input},
            {"role": "assistant", "text": res}
        ]

