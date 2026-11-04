# -*- coding: utf-8 -*-

"""
StudyBuddy AI – Your Adaptive Learning Companion
=================================================
A Flask web application powered by the Groq API that provides:
  1. Resource Fusion & Content Agent  – summarises study materials, creates flashcards & concept maps
  2. Predictive Study Planner Agent   – builds personalised study plans with smart alerts

Run:  python app.py
URL:  http://127.0.0.1:5000
"""

import os
import json
import re
from flask import Flask, request, jsonify, render_template_string, session
from groq import Groq
from dotenv import load_dotenv

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.3-70b-versatile")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "studybuddy-secret-2024")

# ── Groq client ───────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    print("\n⚠️  WARNING: GROQ_API_KEY is not set!")
    print("   Create a free key at https://console.groq.com")
    print("   Then add it to a .env file:  GROQ_API_KEY=gsk_...\n")

client = Groq(api_key=GROQ_API_KEY if GROQ_API_KEY else "invalid-key-please-set-env")


# ─────────────────────────────────────────────────────────────────────────────
#  CORE AI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """
    Call the Groq API and return the assistant's text reply.
    Raises a ValueError with a friendly message on API / network errors.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "No Groq API key found. Please create a free key at https://console.groq.com "
            "and add it to a .env file as: GROQ_API_KEY=gsk_..."
        )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            model=MODEL_NAME,
            temperature=temperature,
            max_tokens=2048,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as exc:
        error_str = str(exc)
        # Give a clear, actionable message for the most common errors
        if "401" in error_str or "invalid_api_key" in error_str:
            raise ValueError(
                "Invalid Groq API key (401). Please get a free key at "
                "https://console.groq.com and set it in your .env file as GROQ_API_KEY=gsk_..."
            ) from exc
        if "model_not_found" in error_str or "404" in error_str:
            raise ValueError(
                f"Model '{MODEL_NAME}' not found on Groq. "
                "Try setting MODEL_NAME=llama-3.3-70b-versatile in your .env file."
            ) from exc
        if "rate_limit" in error_str or "429" in error_str:
            raise ValueError(
                "Groq rate limit hit. Please wait a moment and try again."
            ) from exc
        raise ValueError(f"Groq API error: {error_str}") from exc


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT 1 – Resource Fusion & Content Agent
# ─────────────────────────────────────────────────────────────────────────────

def resource_fusion_agent(content: str, content_type: str = "text") -> dict:
    """
    Accepts raw study content and returns:
      - summary       : concise topic summary
      - flashcards    : list of {question, answer} dicts
      - concept_map   : text-based concept map
    """
    system_prompt = (
        "You are an expert educational content processor. "
        "Your job is to analyse study material and produce three things:\n"
        "1. A concise summary (3-5 sentences).\n"
        "2. Five flashcard Q&A pairs that test key concepts.\n"
        "3. A text-based concept map showing how the main ideas connect "
        "   (use indentation and arrows like '→' to show relationships).\n\n"
        "Always respond with VALID JSON in exactly this structure:\n"
        "{\n"
        '  "summary": "...",\n'
        '  "flashcards": [\n'
        '    {"question": "...", "answer": "..."},\n'
        "    ...\n"
        "  ],\n"
        '  "concept_map": "..."\n'
        "}"
    )

    source_label = {
        "text":     "text snippet",
        "url":      "YouTube/web URL transcript",
        "ocr":      "OCR-extracted text from an image",
    }.get(content_type, "study material")

    user_prompt = (
        f"Process the following {source_label} and return the JSON response:\n\n"
        f"--- CONTENT START ---\n{content}\n--- CONTENT END ---"
    )

    raw = generate_response(system_prompt, user_prompt, temperature=0.4)

    # Strip markdown code fences if the model wraps the JSON
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: wrap raw text so the UI always gets valid data
        result = {
            "summary": raw[:500] if len(raw) > 500 else raw,
            "flashcards": [{"question": "See full response", "answer": raw}],
            "concept_map": "Concept map not available – raw response returned.",
        }

    # Ensure all expected keys exist
    result.setdefault("summary", "No summary generated.")
    result.setdefault("flashcards", [])
    result.setdefault("concept_map", "")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT 2 – Predictive Study Planner Agent
# ─────────────────────────────────────────────────────────────────────────────

def study_planner_agent(
    topics: list,
    exam_date: str,
    weak_areas: list,
    strong_areas: list,
    daily_hours: float,
) -> dict:
    """
    Generates a personalised study plan with:
      - daily_plan    : list of {day, tasks[]} dicts (7 days)
      - alerts        : list of actionable recommendation strings
      - weekly_summary: brief overview paragraph
    """
    system_prompt = (
        "You are an expert academic coach and study strategist. "
        "Given a student's syllabus, exam date, weak areas, strong areas, "
        "and available daily study hours, you must create a realistic, personalised "
        "7-day study plan.\n\n"
        "Always respond with VALID JSON in exactly this structure:\n"
        "{\n"
        '  "weekly_summary": "...",\n'
        '  "daily_plan": [\n'
        '    {"day": "Day 1 – Mon", "tasks": ["Task 1", "Task 2"]},\n'
        "    ...\n"
        "  ],\n"
        '  "alerts": [\n'
        '    "Alert / recommendation 1",\n'
        "    ...\n"
        "  ]\n"
        "}"
    )

    user_prompt = (
        f"Create a personalised 7-day study plan with the following details:\n\n"
        f"Syllabus Topics  : {', '.join(topics)}\n"
        f"Exam Date        : {exam_date}\n"
        f"Weak Areas       : {', '.join(weak_areas) if weak_areas else 'None specified'}\n"
        f"Strong Areas     : {', '.join(strong_areas) if strong_areas else 'None specified'}\n"
        f"Daily Study Hours: {daily_hours} hours\n\n"
        "Make sure the weak areas receive extra attention. "
        "Include specific, actionable tasks for each day. "
        "Alerts should warn about high-priority or frequently tested topics."
    )

    raw = generate_response(system_prompt, user_prompt, temperature=0.5)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "weekly_summary": raw[:600] if len(raw) > 600 else raw,
            "daily_plan": [],
            "alerts": ["Raw response returned – JSON parse failed."],
        }

    result.setdefault("weekly_summary", "")
    result.setdefault("daily_plan", [])
    result.setdefault("alerts", [])
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR – routes requests to the correct agent
# ─────────────────────────────────────────────────────────────────────────────

def orchestrate(intent: str, payload: dict) -> dict:
    """
    Simple intent-based router.
    intent: 'summarise' | 'plan'
    """
    if intent == "summarise":
        content      = payload.get("content", "").strip()
        content_type = payload.get("content_type", "text")
        if not content:
            return {"error": "No content provided."}
        return resource_fusion_agent(content, content_type)

    elif intent == "plan":
        topics       = [t.strip() for t in payload.get("topics", "").split(",") if t.strip()]
        exam_date    = payload.get("exam_date", "Not specified")
        weak_areas   = [t.strip() for t in payload.get("weak_areas",   "").split(",") if t.strip()]
        strong_areas = [t.strip() for t in payload.get("strong_areas", "").split(",") if t.strip()]
        daily_hours  = float(payload.get("daily_hours", 2))
        if not topics:
            return {"error": "Please provide at least one syllabus topic."}
        return study_planner_agent(topics, exam_date, weak_areas, strong_areas, daily_hours)

    return {"error": f"Unknown intent: {intent}"}


# ─────────────────────────────────────────────────────────────────────────────
#  HTML TEMPLATES  (all inline via render_template_string)
# ─────────────────────────────────────────────────────────────────────────────

BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>StudyBuddy AI – {{ page_title }}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
  <style>
    :root {
      --sb-primary:   #4f46e5;
      --sb-secondary: #7c3aed;
      --sb-accent:    #06b6d4;
      --sb-dark:      #0f172a;
      --sb-surface:   #1e293b;
      --sb-card:      #162032;
    }
    body { background: var(--sb-dark); color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

    /* ── Sidebar ── */
    #sidebar {
      min-height: 100vh; width: 250px; background: var(--sb-surface);
      border-right: 1px solid #334155; position: fixed; top: 0; left: 0;
      display: flex; flex-direction: column; z-index: 100;
    }
    #sidebar .brand { padding: 1.25rem 1rem; border-bottom: 1px solid #334155; }
    #sidebar .brand h5 { color: #fff; font-weight: 700; margin: 0; }
    #sidebar .brand small { color: var(--sb-accent); font-size: .7rem; letter-spacing: .05em; text-transform: uppercase; }
    #sidebar .nav-link {
      color: #94a3b8; padding: .6rem 1rem; border-radius: 8px;
      margin: 2px 8px; display: flex; align-items: center; gap: .5rem;
      transition: background .2s, color .2s;
    }
    #sidebar .nav-link:hover, #sidebar .nav-link.active {
      background: rgba(79,70,229,.2); color: #fff;
    }
    #sidebar .nav-link i { font-size: 1.1rem; }
    #sidebar .sidebar-footer { margin-top: auto; padding: 1rem; border-top: 1px solid #334155; font-size: .75rem; color: #64748b; }

    /* ── Main content ── */
    #main { margin-left: 250px; padding: 1.75rem; }

    /* ── Cards ── */
    .sb-card {
      background: var(--sb-card); border: 1px solid #334155;
      border-radius: 12px; padding: 1.5rem;
    }
    .sb-card-header { font-size: 1rem; font-weight: 600; color: #cbd5e1; margin-bottom: 1rem; display: flex; align-items: center; gap: .5rem; }

    /* ── Flashcard flip ── */
    .flashcard-wrap { perspective: 900px; height: 180px; cursor: pointer; }
    .flashcard-inner { position: relative; width: 100%; height: 100%; transition: transform .5s; transform-style: preserve-3d; }
    .flashcard-wrap.flipped .flashcard-inner { transform: rotateY(180deg); }
    .flashcard-front, .flashcard-back {
      position: absolute; inset: 0; backface-visibility: hidden;
      border-radius: 12px; padding: 1.25rem; display: flex; flex-direction: column;
      justify-content: center; align-items: center; text-align: center;
    }
    .flashcard-front { background: linear-gradient(135deg,#4f46e5,#7c3aed); color: #fff; }
    .flashcard-back  { background: linear-gradient(135deg,#0891b2,#06b6d4); color: #fff; transform: rotateY(180deg); }
    .flashcard-label { font-size: .65rem; text-transform: uppercase; letter-spacing: .08em; opacity: .75; margin-bottom: .4rem; }

    /* ── Concept map ── */
    .concept-map { font-family: 'Courier New', monospace; font-size: .85rem; white-space: pre-wrap; color: #a5f3fc; }

    /* ── Progress & alerts ── */
    .progress { background: #1e293b; }
    .alert-card { border-left: 4px solid var(--sb-accent); background: rgba(6,182,212,.07); border-radius: 8px; padding: .75rem 1rem; margin-bottom: .5rem; font-size: .9rem; }

    /* ── Stat boxes ── */
    .stat-box { background: var(--sb-card); border: 1px solid #334155; border-radius: 12px; padding: 1.25rem; text-align: center; }
    .stat-box .number { font-size: 2rem; font-weight: 700; color: var(--sb-accent); }
    .stat-box .label  { font-size: .8rem; color: #94a3b8; margin-top: .25rem; }

    /* ── Loader overlay ── */
    #loader {
      display: none; position: fixed; inset: 0; background: rgba(15,23,42,.8);
      z-index: 9999; justify-content: center; align-items: center; flex-direction: column;
    }
    #loader.show { display: flex; }
    .spinner-ring { width: 56px; height: 56px; border: 5px solid #334155; border-top-color: var(--sb-primary); border-radius: 50%; animation: spin .8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Miscellaneous ── */
    .page-header { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-bottom: .25rem; }
    .page-sub    { color: #64748b; font-size: .9rem; margin-bottom: 1.5rem; }
    .gradient-text { background: linear-gradient(90deg, var(--sb-primary), var(--sb-accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .btn-primary-sb { background: var(--sb-primary); border: none; color: #fff; border-radius: 8px; padding: .55rem 1.4rem; font-weight: 600; transition: opacity .2s; }
    .btn-primary-sb:hover { opacity: .85; color: #fff; }
    textarea.form-control, input.form-control, select.form-control {
      background: #1e293b; border: 1px solid #334155; color: #e2e8f0;
    }
    textarea.form-control:focus, input.form-control:focus, select.form-control:focus {
      background: #1e293b; border-color: var(--sb-primary); color: #e2e8f0; box-shadow: 0 0 0 3px rgba(79,70,229,.25);
    }
    label { color: #94a3b8; font-size: .875rem; }
    ::placeholder { color: #475569 !important; }
    .day-card { background: #0f172a; border: 1px solid #1e3a5f; border-radius: 10px; padding: 1rem; }
    .day-card h6 { color: var(--sb-accent); font-weight: 600; margin-bottom: .6rem; }
    .day-card li { font-size: .875rem; color: #cbd5e1; margin-bottom: .3rem; }
    @media (max-width: 768px) {
      #sidebar { transform: translateX(-100%); transition: transform .3s; }
      #sidebar.open { transform: translateX(0); }
      #main { margin-left: 0; padding: 1rem; }
      #toggle-sidebar { display: block !important; }
    }
  </style>
</head>
<body>

<!-- ── Loading Overlay ── -->
<div id="loader">
  <div class="spinner-ring mb-3"></div>
  <p class="text-slate-400" id="loader-msg">AI is thinking…</p>
</div>

<!-- ── Sidebar ── -->
<nav id="sidebar">
  <div class="brand">
    <div class="d-flex align-items-center gap-2">
      <i class="bi bi-mortarboard-fill fs-4" style="color:var(--sb-accent)"></i>
      <div>
        <h5>StudyBuddy AI</h5>
        <small>Adaptive Learning</small>
      </div>
    </div>
  </div>
  <div class="py-2">
    <div class="px-3 pb-1 pt-2" style="font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:#475569;">Navigation</div>
    <a href="/"         class="nav-link {% if active=='home'    %}active{% endif %}"><i class="bi bi-house-door"></i> Dashboard</a>
    <a href="/summarise" class="nav-link {% if active=='summarise' %}active{% endif %}"><i class="bi bi-file-earmark-text"></i> Upload &amp; Summarise</a>
    <a href="/planner"  class="nav-link {% if active=='planner' %}active{% endif %}"><i class="bi bi-calendar-check"></i> Study Planner</a>
    <a href="/about"    class="nav-link {% if active=='about'   %}active{% endif %}"><i class="bi bi-info-circle"></i> About</a>
  </div>
  <div class="sidebar-footer">
    Powered by <strong style="color:#a5b4fc">Groq API</strong><br/>
    Model: <code style="color:#6ee7b7;font-size:.7rem">{{ model }}</code>
  </div>
</nav>

<!-- ── Mobile toggle ── -->
<button id="toggle-sidebar" class="btn btn-sm d-none" style="position:fixed;top:12px;left:12px;z-index:200;background:var(--sb-surface);color:#fff;border:1px solid #334155;" onclick="document.getElementById('sidebar').classList.toggle('open')">
  <i class="bi bi-list fs-5"></i>
</button>

<!-- ── Main ── -->
<main id="main">
  {{ content | safe }}
</main>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
  // ── Loader helpers ──
  function showLoader(msg='AI is thinking…'){
    document.getElementById('loader-msg').textContent = msg;
    document.getElementById('loader').classList.add('show');
  }
  function hideLoader(){ document.getElementById('loader').classList.remove('show'); }

  // ── Flashcard flip ──
  document.querySelectorAll('.flashcard-wrap').forEach(card => {
    card.addEventListener('click', () => card.classList.toggle('flipped'));
  });
</script>
{{ extra_js | safe }}
</body>
</html>
"""


def render_page(page_title, active, content, extra_js=""):
    return render_template_string(
        BASE_HTML,
        page_title=page_title,
        active=active,
        content=content,
        extra_js=extra_js,
        model=MODEL_NAME,
    )


# ─── HOME / DASHBOARD ────────────────────────────────────────────────────────

HOME_CONTENT = """
<div class="page-header gradient-text">🎓 Welcome to StudyBuddy AI</div>
<p class="page-sub">Your adaptive AI-powered learning companion for smarter, faster exam prep.</p>

<!-- Stat row -->
<div class="row g-3 mb-4">
  <div class="col-6 col-md-3"><div class="stat-box"><div class="number">2</div><div class="label">AI Agents</div></div></div>
  <div class="col-6 col-md-3"><div class="stat-box"><div class="number">∞</div><div class="label">Study Topics</div></div></div>
  <div class="col-6 col-md-3"><div class="stat-box"><div class="number">7</div><div class="label">Day Planner</div></div></div>
  <div class="col-6 col-md-3"><div class="stat-box"><div class="number">100%</div><div class="label">Free to Use</div></div></div>
</div>

<!-- Feature cards -->
<div class="row g-3 mb-4">
  <div class="col-md-6">
    <div class="sb-card h-100">
      <div class="sb-card-header"><i class="bi bi-file-earmark-richtext text-indigo-400" style="color:#818cf8"></i> Resource Fusion Agent</div>
      <p style="color:#94a3b8;font-size:.9rem">Upload text snippets, paste a YouTube transcript, or enter OCR text from an image. The agent instantly generates a concise summary, flashcard Q&amp;A pairs, and an interactive concept map.</p>
      <a href="/summarise" class="btn btn-primary-sb btn-sm mt-2">Try it →</a>
    </div>
  </div>
  <div class="col-md-6">
    <div class="sb-card h-100">
      <div class="sb-card-header"><i class="bi bi-calendar4-range" style="color:#34d399"></i> Predictive Study Planner</div>
      <p style="color:#94a3b8;font-size:.9rem">Enter your syllabus, exam date, weak &amp; strong areas, and daily study hours. Get a personalised 7-day plan with smart alerts highlighting high-priority and weak-area topics.</p>
      <a href="/planner" class="btn btn-primary-sb btn-sm mt-2">Plan Now →</a>
    </div>
  </div>
</div>

<!-- How it works -->
<div class="sb-card">
  <div class="sb-card-header"><i class="bi bi-lightning-charge" style="color:#fbbf24"></i> How It Works</div>
  <div class="row g-3 text-center">
    <div class="col-md-4">
      <div style="font-size:2rem;margin-bottom:.5rem">📤</div>
      <div style="font-weight:600;color:#e2e8f0">1. Provide Content</div>
      <div style="font-size:.85rem;color:#64748b">Paste notes, a URL, or OCR text</div>
    </div>
    <div class="col-md-4">
      <div style="font-size:2rem;margin-bottom:.5rem">🤖</div>
      <div style="font-weight:600;color:#e2e8f0">2. AI Processes</div>
      <div style="font-size:.85rem;color:#64748b">Groq LLM analyses &amp; structures knowledge</div>
    </div>
    <div class="col-md-4">
      <div style="font-size:2rem;margin-bottom:.5rem">📚</div>
      <div style="font-weight:600;color:#e2e8f0">3. Study Smarter</div>
      <div style="font-size:.85rem;color:#64748b">Use flashcards, maps &amp; daily plans</div>
    </div>
  </div>
</div>
"""


@app.route("/")
def home():
    return render_page("Dashboard", "home", HOME_CONTENT)


# ─── SUMMARISE PAGE ───────────────────────────────────────────────────────────

SUMMARISE_CONTENT = """
<div class="page-header gradient-text"><i class="bi bi-file-earmark-text"></i> Upload &amp; Summarise</div>
<p class="page-sub">Paste study material below. The Resource Fusion Agent will generate a summary, flashcards, and a concept map.</p>

<div class="row g-3">
  <!-- Input panel -->
  <div class="col-lg-5">
    <div class="sb-card">
      <div class="sb-card-header"><i class="bi bi-upload" style="color:#818cf8"></i> Input Content</div>
      <div class="mb-3">
        <label class="mb-1">Content Type</label>
        <select id="contentType" class="form-control form-select" style="background:#1e293b;border-color:#334155;color:#e2e8f0">
          <option value="text">📝 Text Snippet / Notes</option>
          <option value="url">🔗 YouTube / Web Transcript (paste text)</option>
          <option value="ocr">📷 OCR Text (from image)</option>
        </select>
      </div>
      <div class="mb-3">
        <label class="mb-1">Study Material</label>
        <textarea id="studyContent" class="form-control" rows="10"
          placeholder="Paste your notes, transcript, or OCR text here…"></textarea>
      </div>
      <button class="btn btn-primary-sb w-100" onclick="runSummarise()">
        <i class="bi bi-cpu me-1"></i> Generate with AI
      </button>
    </div>
  </div>

  <!-- Results panel -->
  <div class="col-lg-7">
    <!-- Summary -->
    <div class="sb-card mb-3" id="summaryCard" style="display:none">
      <div class="sb-card-header"><i class="bi bi-card-text" style="color:#34d399"></i> Topic Summary</div>
      <p id="summaryText" style="color:#cbd5e1;line-height:1.7;font-size:.925rem"></p>
    </div>

    <!-- Flashcards -->
    <div class="sb-card mb-3" id="flashcardsCard" style="display:none">
      <div class="sb-card-header"><i class="bi bi-stack" style="color:#fbbf24"></i> Flashcards <small style="color:#475569;font-weight:400;margin-left:.4rem">(click to flip)</small></div>
      <div class="row g-3" id="flashcardsContainer"></div>
    </div>

    <!-- Concept Map -->
    <div class="sb-card" id="conceptCard" style="display:none">
      <div class="sb-card-header"><i class="bi bi-diagram-3" style="color:#f472b6"></i> Concept Map</div>
      <pre class="concept-map" id="conceptMap"></pre>
    </div>
  </div>
</div>
"""

SUMMARISE_JS = """
<script>
async function runSummarise() {
  const content = document.getElementById('studyContent').value.trim();
  const contentType = document.getElementById('contentType').value;
  if (!content) { alert('Please enter some study content first.'); return; }

  showLoader('Resource Fusion Agent is processing…');

  try {
    const resp = await fetch('/api/summarise', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ content, content_type: contentType })
    });
    const data = await resp.json();
    hideLoader();

    if (data.error) { alert('Error: ' + data.error); return; }

    // Summary
    document.getElementById('summaryText').textContent = data.summary || '';
    document.getElementById('summaryCard').style.display = 'block';

    // Flashcards
    const fc = data.flashcards || [];
    const fcContainer = document.getElementById('flashcardsContainer');
    fcContainer.innerHTML = '';
    fc.forEach((card, i) => {
      fcContainer.innerHTML += `
        <div class="col-md-6">
          <div class="flashcard-wrap" id="fc${i}">
            <div class="flashcard-inner">
              <div class="flashcard-front">
                <div class="flashcard-label">Question</div>
                <div>${card.question}</div>
              </div>
              <div class="flashcard-back">
                <div class="flashcard-label">Answer</div>
                <div>${card.answer}</div>
              </div>
            </div>
          </div>
        </div>`;
    });
    document.getElementById('flashcardsCard').style.display = fc.length ? 'block' : 'none';
    // Re-bind flip on newly created cards
    document.querySelectorAll('.flashcard-wrap').forEach(c => {
      c.addEventListener('click', () => c.classList.toggle('flipped'));
    });

    // Concept map
    document.getElementById('conceptMap').textContent = data.concept_map || '';
    document.getElementById('conceptCard').style.display = data.concept_map ? 'block' : 'none';

  } catch(e) {
    hideLoader();
    alert('Network or server error: ' + e.message);
  }
}
</script>
"""


@app.route("/summarise")
def summarise_page():
    return render_page("Upload & Summarise", "summarise", SUMMARISE_CONTENT, SUMMARISE_JS)


@app.route("/api/summarise", methods=["POST"])
def api_summarise():
    data = request.get_json(silent=True) or {}
    if not data.get("content", "").strip():
        return jsonify({"error": "No content provided."}), 400
    try:
        result = orchestrate("summarise", data)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 502


# ─── PLANNER PAGE ─────────────────────────────────────────────────────────────

PLANNER_CONTENT = """
<div class="page-header gradient-text"><i class="bi bi-calendar-check"></i> Predictive Study Planner</div>
<p class="page-sub">Fill in your study profile and let the AI build a personalised 7-day plan with smart alerts.</p>

<div class="row g-3">
  <!-- Form panel -->
  <div class="col-lg-4">
    <div class="sb-card">
      <div class="sb-card-header"><i class="bi bi-person-badge" style="color:#818cf8"></i> Study Profile</div>
      <div class="mb-3">
        <label class="mb-1">Syllabus Topics <span style="color:#ef4444">*</span></label>
        <input id="topics" class="form-control" placeholder="e.g. Machine Learning, Neural Networks, SVM" />
        <small style="color:#475569">Comma-separated list</small>
      </div>
      <div class="mb-3">
        <label class="mb-1">Exam Date</label>
        <input id="examDate" type="date" class="form-control" />
      </div>
      <div class="mb-3">
        <label class="mb-1">Weak Areas</label>
        <input id="weakAreas" class="form-control" placeholder="e.g. Backpropagation, Gradient Descent" />
      </div>
      <div class="mb-3">
        <label class="mb-1">Strong Areas</label>
        <input id="strongAreas" class="form-control" placeholder="e.g. Linear Regression, Decision Trees" />
      </div>
      <div class="mb-3">
        <label class="mb-1">Daily Study Hours: <span id="hoursLabel" style="color:var(--sb-accent)">3</span>h</label>
        <input type="range" id="dailyHours" class="form-range" min="1" max="10" value="3"
          oninput="document.getElementById('hoursLabel').textContent=this.value" style="accent-color:var(--sb-primary)"/>
      </div>
      <button class="btn btn-primary-sb w-100" onclick="runPlanner()">
        <i class="bi bi-cpu me-1"></i> Generate Study Plan
      </button>
    </div>
  </div>

  <!-- Results panel -->
  <div class="col-lg-8">
    <!-- Weekly summary + progress bar -->
    <div class="sb-card mb-3" id="summarySection" style="display:none">
      <div class="sb-card-header"><i class="bi bi-bar-chart-line" style="color:#34d399"></i> Plan Overview</div>
      <p id="weeklySummary" style="color:#cbd5e1;font-size:.925rem;line-height:1.7"></p>
      <div class="mb-1 d-flex justify-content-between"><small style="color:#94a3b8">Weekly Prep Progress</small><small style="color:#94a3b8" id="progLabel">0%</small></div>
      <div class="progress" style="height:8px;border-radius:4px">
        <div id="progressBar" class="progress-bar" role="progressbar" style="width:0%;background:linear-gradient(90deg,#4f46e5,#06b6d4);border-radius:4px" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
      </div>
    </div>

    <!-- Alerts -->
    <div id="alertsSection" style="display:none" class="mb-3">
      <div class="sb-card-header"><i class="bi bi-exclamation-triangle" style="color:#f59e0b"></i> Smart Alerts &amp; Recommendations</div>
      <div id="alertsList"></div>
    </div>

    <!-- Daily plan grid -->
    <div class="sb-card" id="planSection" style="display:none">
      <div class="sb-card-header"><i class="bi bi-calendar3" style="color:#a78bfa"></i> 7-Day Daily Plan</div>
      <div class="row g-2" id="dailyPlanGrid"></div>
    </div>
  </div>
</div>
"""

PLANNER_JS = """
<script>
async function runPlanner() {
  const topics     = document.getElementById('topics').value.trim();
  const examDate   = document.getElementById('examDate').value;
  const weakAreas  = document.getElementById('weakAreas').value.trim();
  const strongAreas= document.getElementById('strongAreas').value.trim();
  const dailyHours = document.getElementById('dailyHours').value;

  if (!topics) { alert('Please enter at least one syllabus topic.'); return; }

  showLoader('Study Planner Agent is building your plan…');

  try {
    const resp = await fetch('/api/plan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ topics, exam_date: examDate || 'Not specified', weak_areas: weakAreas, strong_areas: strongAreas, daily_hours: dailyHours })
    });
    const data = await resp.json();
    hideLoader();

    if (data.error) { alert('Error: ' + data.error); return; }

    // Weekly summary
    document.getElementById('weeklySummary').textContent = data.weekly_summary || '';
    document.getElementById('summarySection').style.display = 'block';

    // Animate progress bar
    const totalDays = (data.daily_plan || []).length;
    const pct = totalDays > 0 ? Math.round((totalDays / 7) * 100) : 0;
    const bar = document.getElementById('progressBar');
    bar.style.width = pct + '%';
    bar.setAttribute('aria-valuenow', pct);
    document.getElementById('progLabel').textContent = pct + '%';

    // Alerts
    const alerts = data.alerts || [];
    const alertsList = document.getElementById('alertsList');
    alertsList.innerHTML = '';
    alerts.forEach(a => {
      alertsList.innerHTML += `<div class="alert-card"><i class="bi bi-lightbulb-fill me-2" style="color:#fbbf24"></i>${a}</div>`;
    });
    document.getElementById('alertsSection').style.display = alerts.length ? 'block' : 'none';

    // Daily plan
    const plan = data.daily_plan || [];
    const grid = document.getElementById('dailyPlanGrid');
    grid.innerHTML = '';
    plan.forEach(day => {
      const tasks = (day.tasks || []).map(t => `<li>${t}</li>`).join('');
      grid.innerHTML += `
        <div class="col-md-6 col-lg-4">
          <div class="day-card">
            <h6>${day.day || 'Day'}</h6>
            <ul class="ps-3 mb-0">${tasks}</ul>
          </div>
        </div>`;
    });
    document.getElementById('planSection').style.display = plan.length ? 'block' : 'none';

  } catch(e) {
    hideLoader();
    alert('Network or server error: ' + e.message);
  }
}
</script>
"""


@app.route("/planner")
def planner_page():
    return render_page("Study Planner", "planner", PLANNER_CONTENT, PLANNER_JS)


@app.route("/api/plan", methods=["POST"])
def api_plan():
    data = request.get_json(silent=True) or {}
    if not data.get("topics", "").strip():
        return jsonify({"error": "Please provide at least one topic."}), 400
    try:
        result = orchestrate("plan", data)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 502


# ─── ABOUT PAGE ───────────────────────────────────────────────────────────────

ABOUT_CONTENT = """
<div class="page-header gradient-text"><i class="bi bi-info-circle"></i> About StudyBuddy AI</div>
<p class="page-sub">An open-source AI-powered learning companion built for students and hackathons.</p>

<div class="row g-3">
  <div class="col-md-8">
    <div class="sb-card mb-3">
      <div class="sb-card-header"><i class="bi bi-layers" style="color:#818cf8"></i> Architecture</div>
      <p style="color:#94a3b8;font-size:.9rem;line-height:1.8">
        StudyBuddy AI is a single-file Flask application (<code style="color:#6ee7b7">app.py</code>) that uses
        <strong style="color:#e2e8f0">two AI agents</strong> orchestrated by a lightweight router:
      </p>
      <ul style="color:#94a3b8;font-size:.9rem;line-height:2">
        <li><strong style="color:#a5b4fc">Resource Fusion Agent</strong> – processes study material (text, simulated OCR, or web/YouTube transcripts) and returns a structured summary, flashcards, and a concept map.</li>
        <li><strong style="color:#6ee7b7">Predictive Study Planner Agent</strong> – takes syllabus topics, exam date, and skill levels to output a 7-day study plan with smart priority alerts.</li>
      </ul>
    </div>

    <div class="sb-card mb-3">
      <div class="sb-card-header"><i class="bi bi-tools" style="color:#34d399"></i> Tech Stack</div>
      <div class="row g-2">
        <div class="col-sm-6">
          <ul style="color:#94a3b8;font-size:.9rem;line-height:2;padding-left:1.25rem">
            <li>Python 3.10+</li>
            <li>Flask 3.x</li>
            <li>Groq Python SDK</li>
            <li>python-dotenv</li>
          </ul>
        </div>
        <div class="col-sm-6">
          <ul style="color:#94a3b8;font-size:.9rem;line-height:2;padding-left:1.25rem">
            <li>Bootstrap 5.3</li>
            <li>Bootstrap Icons 1.11</li>
            <li>Vanilla JavaScript (Fetch API)</li>
            <li>CSS Flip Animations</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="sb-card">
      <div class="sb-card-header"><i class="bi bi-code-slash" style="color:#f472b6"></i> Quick Setup</div>
      <pre style="background:#0f172a;border-radius:8px;padding:1rem;color:#a5f3fc;font-size:.82rem;overflow-x:auto">
# 1. Clone / download
git clone &lt;repo-url&gt; &amp;&amp; cd StudyBuddy

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run
python app.py
# Open http://127.0.0.1:5000</pre>
    </div>
  </div>

  <div class="col-md-4">
    <div class="sb-card mb-3">
      <div class="sb-card-header"><i class="bi bi-lightning-charge" style="color:#fbbf24"></i> Key Features</div>
      <ul style="color:#94a3b8;font-size:.875rem;line-height:2.2;padding-left:1.25rem">
        <li>✅ Two AI agents with orchestration</li>
        <li>✅ Flashcard flip animation</li>
        <li>✅ Text-based concept maps</li>
        <li>✅ 7-day personalised plans</li>
        <li>✅ Smart priority alerts</li>
        <li>✅ Dark mode dashboard</li>
        <li>✅ Mobile responsive sidebar</li>
        <li>✅ Loading overlays</li>
        <li>✅ Input validation</li>
        <li>✅ API error handling</li>
        <li>✅ Single-file deployment</li>
        <li>✅ 100% free &amp; open source</li>
      </ul>
    </div>

    <div class="sb-card">
      <div class="sb-card-header"><i class="bi bi-mortarboard" style="color:#a78bfa"></i> Built For</div>
      <p style="color:#94a3b8;font-size:.875rem;line-height:1.8">
        AI/ML Hackathons, College Demonstrations, Personal Study, and anyone who wants a free, offline-capable study assistant powered by open-source LLMs via the Groq API.
      </p>
    </div>
  </div>
</div>
"""


@app.route("/about")
def about_page():
    return render_page("About", "about", ABOUT_CONTENT)


# ─────────────────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  StudyBuddy AI – Adaptive Learning Companion")
    print("  URL  : http://127.0.0.1:5000")
    print(f"  Model: {MODEL_NAME}")
    if not GROQ_API_KEY:
        print("  ⚠️  GROQ_API_KEY not set – get one free at:")
        print("      https://console.groq.com")
        print("      Then add to .env:  GROQ_API_KEY=gsk_...")
    else:
        print("  ✅  API key loaded")
    print("=" * 55)
    app.run(debug=True, port=5000)
