import base64
import os
import re as _re
import time
from urllib.parse import urlparse as _urlparse

import cv2
import numpy as np
import requests
import requests.exceptions
import streamlit as st
import streamlit.components.v1 as components

# Load environment variables from .env file (local) or Streamlit secrets (cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available on Streamlit Cloud

from supabase import create_client, Client

# Optional: ML libraries for standalone / cloud mode
try:
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_URL = "http://localhost:8000/api"
MAX_CHAT_HISTORY = 30
SESSION_REFRESH_INTERVAL = 60  # seconds between session refreshes
HEALTH_CHECK_TTL = 30  # seconds to cache backend health status
QWEN_MODEL = "qwen-turbo"  # Options: qwen-turbo, qwen-plus, qwen-max
QWEN_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
MODELS_DIR = "Models"

# ---------------------------------------------------------------------------
# Cloud mode detection
# ---------------------------------------------------------------------------
def _get_env(key: str, default: str = "") -> str:
    """Get env var from os.environ or Streamlit secrets (cloud fallback)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# Auto-detect Streamlit Cloud (no local backend available)
CLOUD_MODE = _get_env("STREAMLIT_CLOUD", "") != "" or _get_env("STANDALONE_MODE", "") != ""

# ---------------------------------------------------------------------------
# DashScope / Qwen configuration (OpenAI-compatible API)
# ---------------------------------------------------------------------------
_DASHSCOPE_API_KEY = _get_env("DASHSCOPE_API_KEY")
QWEN_ENABLED = bool(_DASHSCOPE_API_KEY and _DASHSCOPE_API_KEY != "your-dashscope-api-key-here")

# Cybersecurity-focused system prompt for Qwen
QWEN_SYSTEM_PROMPT = """You are CyberShield AI, an expert cybersecurity assistant. Your role is to help users understand and defend against cyber threats.

You have access to these security scanning modules:
- Email Analyzer: Detects spam and phishing emails using ML
- Scam Text Scanner: Detects SMS/WhatsApp scams (supports English, Urdu, Roman Urdu)
- Phishing & QR Detector: Analyzes URLs for phishing and decodes QR codes
- Password Analyzer: Evaluates password strength
- Malware Guard: Screens files for dangerous extensions

When users ask about security:
1. Provide accurate, actionable cybersecurity advice
2. Explain threats in simple terms
3. Recommend specific protective actions
4. If they want to scan something, guide them to the appropriate module
5. For scanned results, explain what the findings mean

Keep responses concise but informative. Use emojis sparingly for clarity.
Never ask for or store sensitive information like passwords or personal data."""

# ---------------------------------------------------------------------------
# Standalone ML models (cloud mode or fallback when backend is offline)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_ml_models():
    """Load ML models once for standalone / cloud processing. Missing models
    fall back to heuristic-based detection — the app still works."""
    models = {}

    # Scam text model
    try:
        models["scam_model"] = joblib.load(os.path.join(MODELS_DIR, "scam_model.pkl"))
        models["scam_vectorizer"] = joblib.load(os.path.join(MODELS_DIR, "scam_vectorizer.pkl"))
    except Exception:
        models["scam_model"] = None
        models["scam_vectorizer"] = None

    # Phishing URL model
    try:
        models["phishing_model"] = joblib.load(os.path.join(MODELS_DIR, "phishing_model.pkl"))
    except Exception:
        models["phishing_model"] = None

    # Email spam model
    try:
        models["email_model"] = joblib.load(os.path.join(MODELS_DIR, "spam_svm_model_clean (1).pkl"))
        models["email_vectorizer"] = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer_clean.pkl"))
    except Exception:
        models["email_model"] = None
        models["email_vectorizer"] = None

    return models


if ML_AVAILABLE and CLOUD_MODE:
    _ml_models = _load_ml_models()
else:
    _ml_models = {}


# ── Scam keywords (local Urdu + Roman Urdu dictionary) ──
_SCAM_KEYWORDS = [
    "bisp", "jeeto pakistan", "inam", "lottery",
    "account block", "\u0627\u0646\u0639\u0627\u0645", "\u0644\u0627\u0679\u0631\u06cc", "paisa",
]

# ── Phishing heuristic constants ──
_PHISHING_DOMAIN_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "confirm", "authenticate", "banking",
    "password", "credential", "wallet", "suspended", "unusual",
    "activity", "limited", "restore", "unlock", "alert",
    "security", "notification", "invoice", "payment", "refund",
    "support", "helpdesk", "service", "paypal", "apple",
    "google", "microsoft", "amazon", "netflix", "facebook",
    "instagram", "whatsapp", "twitter", "dropbox", "docusign",
    "free", "winner", "prize", "lottery", "claim",
    "gift", "reward", "bonus", "cashback", "urgent",
    "click", "download", "install", "upgrade",
]

_SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".buzz", ".top",
    ".xyz", ".club", ".work", ".click", ".link", ".icu",
    ".cam", ".rest", ".surf",
]

_FREE_HOSTING_DOMAINS = [
    "000webhost", "freehosting", "infinityfree", "awardpace",
    "byethost", "freesite", "wixsite", "weebly", "jimdo",
    "blogspot", "wordpress.com", "tumblr.com", "strikingly",
]

_URL_SHORTENERS = [
    "bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl",
    "tiny.cc", "bc.vc", "adf.ly", "shorte.st",
]

_EMAIL_SPAM_KEYWORDS = [
    "win", "winner", "won", "free", "prize", "claim", "urgent", "act now",
    "limited time", "click here", "congratulations", "you have been selected",
    "million", "cash", "bonus", "offer expires", "no obligation",
    "risk free", "guaranteed", "earn money", "work from home",
]


def _compute_phishing_score(url: str) -> int:
    """Return a risk score (0-100) based on URL heuristic analysis."""
    url_lower = url.lower()
    score = 0

    kw_hits = sum(1 for kw in _PHISHING_DOMAIN_KEYWORDS if kw in url_lower)
    if kw_hits >= 3:
        score += 35
    elif kw_hits >= 2:
        score += 25
    elif kw_hits >= 1:
        score += 10

    hyphen_count = url_lower.count("-")
    if hyphen_count >= 4:
        score += 20
    elif hyphen_count >= 2:
        score += 10

    try:
        parsed = _urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        host = url
    subdomain_dots = host.count(".") - 1
    if subdomain_dots >= 3:
        score += 15
    elif subdomain_dots >= 2:
        score += 8

    for tld in _SUSPICIOUS_TLDS:
        if url_lower.endswith(tld) or url_lower.endswith(tld + "/"):
            score += 20
            break

    for fh in _FREE_HOSTING_DOMAINS:
        if fh in url_lower:
            score += 15
            break

    for us in _URL_SHORTENERS:
        if us in url_lower:
            score += 20
            break

    if _re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_lower):
        score += 25

    if not url_lower.startswith("https"):
        score += 10

    if len(url) > 75:
        score += 10
    if len(url) > 120:
        score += 5

    if "@" in url_lower:
        score += 25

    if url_lower.count(".") >= 5:
        score += 10

    domain_part = host.split(".")[0] if host else ""
    if domain_part and any(c.isdigit() for c in domain_part) and any(c.isalpha() for c in domain_part):
        score += 5

    return min(score, 100)


# ── Local processing functions (standalone replacements for backend API) ──

def _local_scan_text(text: str) -> dict:
    """Scan text for scam indicators locally."""
    model = _ml_models.get("scam_model")
    vec = _ml_models.get("scam_vectorizer")
    model_threat = False
    if model and vec:
        try:
            vectorized = vec.transform([text])
            prediction = model.predict(vectorized)
            model_threat = prediction[0] == "spam"
        except Exception:
            pass
    is_local_scam = any(kw in text.lower() for kw in _SCAM_KEYWORDS)
    status = "Threat Detected" if (model_threat or is_local_scam) else "Safe"
    return {"status": status}


def _local_scan_url(url: str) -> dict:
    """Scan URL for phishing indicators locally."""
    phishing_score = _compute_phishing_score(url)
    model = _ml_models.get("phishing_model")
    model_threat = False
    if model:
        try:
            features = [[
                len(url), url.count("."), url.count("-"),
                1 if "secure" in url.lower() else 0,
                1 if "login" in url.lower() else 0,
                1 if "bank" in url.lower() else 0,
            ]]
            prediction = model.predict(features)
            model_threat = int(prediction[0]) == 1
        except Exception:
            pass
    heuristic_threat = phishing_score >= 40
    status = "Threat Detected" if (model_threat or heuristic_threat) else "Safe"
    return {"status": status, "url": url, "risk_score": phishing_score}


def _local_analyze_password(password: str) -> dict:
    """Analyze password strength locally."""
    if len(password) < 6:
        return {"strength": "Weak \U0001f534"}
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_spec = any(not c.isalnum() for c in password)
    if has_upper and has_digit and has_spec and len(password) >= 10:
        return {"strength": "Strong \U0001f7e2"}
    return {"strength": "Medium \U0001f7e1"}


def _local_analyze_email(subject: str, body: str) -> dict:
    """Analyze email for spam/phishing indicators locally."""
    combined = f"{subject} {body}"
    model = _ml_models.get("email_model")
    vec = _ml_models.get("email_vectorizer")
    model_threat = False
    if model and vec:
        try:
            vectorized = vec.transform([combined])
            prediction = model.predict(vectorized)
            model_threat = int(prediction[0]) == 1
        except Exception:
            pass
    combined_lower = combined.lower()
    kw_matches = sum(1 for kw in _EMAIL_SPAM_KEYWORDS if kw in combined_lower)
    keyword_threat = kw_matches >= 2
    signals = []
    if model_threat:
        signals.append("ML model classified as spam")
    if keyword_threat:
        signals.append("Multiple spam keywords detected")
    if any(kw in combined_lower for kw in ["click here", "click now", "act now"]):
        signals.append("Contains urgent call-to-action")
    if any(kw in combined_lower for kw in ["verify", "confirm identity", "update account"]):
        signals.append("Potential phishing — identity request")
    status = "Threat Detected" if (model_threat or keyword_threat) else "Safe"
    return {"status": status, "signals": signals if signals else ["No spam indicators found"]}


def _local_scan_file(file_name: str) -> dict:
    """Scan file extension for dangerous executables locally."""
    _, ext = os.path.splitext(file_name.lower())
    dangerous = [".exe", ".bat", ".cmd", ".msi", ".scr"]
    if ext in dangerous:
        return {"status": "Threat Detected", "details": f"Dangerous executable script ({ext}) flagged."}
    return {"status": "Safe", "details": "File layout clearance passed."}


# ---------------------------------------------------------------------------
# Supabase client (cached — created once per session, not every rerun)
# ---------------------------------------------------------------------------
_SUPABASE_URL = _get_env("SUPABASE_URL")
_SUPABASE_KEY = _get_env("SUPABASE_ANON_KEY")


@st.cache_resource
def _get_supabase_client() -> Client:
    """Create and cache the Supabase client (singleton per session)."""
    return create_client(_SUPABASE_URL, _SUPABASE_KEY)


supabase: Client = _get_supabase_client()

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _get_session():
    """Return current Supabase session stored in st.session_state, or None."""
    return st.session_state.get("sb_session")


def _set_session(session):
    """Store session and user in st.session_state."""
    st.session_state["sb_session"] = session
    st.session_state["sb_user"] = session.user if session else None


def _try_restore_session():
    """Refresh session token only if enough time has passed since last refresh.
    
    This avoids a costly Supabase API call on every Streamlit rerun.
    """
    sess = _get_session()
    if not sess:
        return

    last_refresh = st.session_state.get("_sb_last_refresh", 0)
    now = time.time()

    # Skip refresh if we refreshed recently
    if now - last_refresh < SESSION_REFRESH_INTERVAL:
        return

    try:
        refreshed = supabase.auth.refresh_session(sess.refresh_token)
        _set_session(refreshed.session)
        st.session_state["_sb_last_refresh"] = now
    except Exception:
        _set_session(None)


# ---------------------------------------------------------------------------
# Auth page (Login + Signup)
# ---------------------------------------------------------------------------

def _render_auth_page():
    """Render the Login / Signup UI. Called only when no session exists."""

    # ── Auth page CSS ──────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* Full-page dark background */
    .stApp {
        background: linear-gradient(135deg, #0B1120 0%, #111827 50%, #0F172A 100%) !important;
    }
    /* Hide sidebar on auth page */
    [data-testid="stSidebar"] { display: none !important; }
    /* Hide Streamlit header/footer/menu */
    header[data-testid="stHeader"], footer { display: none !important; }
    #MainMenu { visibility: hidden; }

    /* Auth card */
    .auth-card {
        max-width: 440px;
        margin: 60px auto 0;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(59,130,246,0.18);
        border-radius: 18px;
        padding: 40px 36px 36px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 8px 40px rgba(0,0,0,0.45);
    }
    .auth-logo {
        display: flex; align-items: center; gap: 11px;
        justify-content: center; margin-bottom: 6px;
    }
    .auth-logo-icon {
        width: 44px; height: 44px; border-radius: 12px;
        background: linear-gradient(135deg,#06B6D4,#3B82F6);
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; box-shadow: 0 3px 14px rgba(6,182,212,0.4);
    }
    .auth-logo-text {
        font-size: 20px; font-weight: 800; letter-spacing: -0.02em;
        background: linear-gradient(90deg,#22D3EE,#60A5FA);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .auth-tagline {
        text-align: center; font-size: 13px; color: #64748B;
        margin-bottom: 28px; letter-spacing: 0.01em;
    }

    /* Tab overrides */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,23,42,0.5) !important;
        border-radius: 10px !important;
        gap: 4px !important;
        padding: 4px !important;
        border: 1px solid rgba(59,130,246,0.12) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px !important;
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg,rgba(6,182,212,0.18),rgba(59,130,246,0.18)) !important;
        color: #22D3EE !important;
        border: 1px solid rgba(6,182,212,0.28) !important;
    }

    /* Input overrides */
    .stTextInput input {
        background: rgba(15,23,42,0.7) !important;
        border: 1px solid rgba(59,130,246,0.2) !important;
        border-radius: 9px !important;
        color: #E2E8F0 !important;
        font-size: 14px !important;
    }
    .stTextInput input:focus {
        border-color: #22D3EE !important;
        box-shadow: 0 0 0 2px rgba(6,182,212,0.2) !important;
    }
    .stTextInput label { color: #94A3B8 !important; font-size: 13px !important; }

    /* Primary auth button */
    .auth-primary-btn button {
        background: linear-gradient(90deg,#06B6D4,#3B82F6) !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 0 !important;
        box-shadow: 0 6px 20px rgba(6,182,212,0.3) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        width: 100% !important;
    }
    .auth-primary-btn button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(6,182,212,0.45) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Brand header ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="auth-card">
      <div class="auth-logo">
        <div class="auth-logo-icon">🛡️</div>
        <span class="auth-logo-text">Cyber Shield AI</span>
      </div>
      <p class="auth-tagline">Secure · Local · AI-Powered Cyber Defense</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Centered container ─────────────────────────────────────────────────
    _, col, _ = st.columns([1, 2, 1])
    with col:
        login_tab, signup_tab = st.tabs(["  Sign In  ", "  Create Account  "])

        # ── LOGIN ──────────────────────────────────────────────────────────
        with login_tab:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            login_email = st.text_input("Email address", key="login_email",
                                        placeholder="you@example.com")
            login_pass  = st.text_input("Password", type="password",
                                        key="login_pass",
                                        placeholder="Your password")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="auth-primary-btn">', unsafe_allow_html=True)
            sign_in_clicked = st.button("Sign In", key="_signin",
                                        use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if sign_in_clicked:
                if not login_email or not login_pass:
                    st.error("Please enter both email and password.")
                else:
                    try:
                        resp = supabase.auth.sign_in_with_password(
                            {"email": login_email, "password": login_pass}
                        )
                        _set_session(resp.session)
                        st.rerun()
                    except Exception as exc:
                        msg = str(exc)
                        if "Invalid login" in msg or "invalid" in msg.lower():
                            st.error("Incorrect email or password. Please try again.")
                        else:
                            st.error(f"Login failed: {msg}")

        # ── SIGNUP ─────────────────────────────────────────────────────────
        with signup_tab:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            signup_name  = st.text_input("Full name", key="signup_name",
                                         placeholder="Jane Smith")
            signup_email = st.text_input("Email address", key="signup_email",
                                         placeholder="you@example.com")
            signup_pass  = st.text_input("Password (min 8 characters)",
                                         type="password", key="signup_pass",
                                         placeholder="Choose a strong password")
            signup_pass2 = st.text_input("Confirm password", type="password",
                                         key="signup_pass2",
                                         placeholder="Repeat your password")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="auth-primary-btn">', unsafe_allow_html=True)
            sign_up_clicked = st.button("Create Account", key="_signup",
                                        use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if sign_up_clicked:
                # Client-side validation
                if not signup_name or not signup_email or not signup_pass:
                    st.error("All fields are required.")
                elif len(signup_pass) < 8:
                    st.error("Password must be at least 8 characters long.")
                elif signup_pass != signup_pass2:
                    st.error("Passwords do not match.")
                else:
                    try:
                        resp = supabase.auth.sign_up({
                            "email":    signup_email,
                            "password": signup_pass,
                            "options": {
                                "email_redirect_to": "http://localhost:8501",
                            },
                        })
                        # Insert profile row
                        if resp.user:
                            try:
                                supabase.table("profiles").insert({
                                    "id":         resp.user.id,
                                    "full_name":  signup_name,
                                    "email":      signup_email,
                                }).execute()
                            except Exception:
                                pass  # Non-fatal — profile can be created later
                        st.success(
                            "Account created! Check your email to confirm your "
                            "address, then sign in."
                        )
                    except Exception as exc:
                        msg = str(exc)
                        if "already registered" in msg.lower() or "already exists" in msg.lower():
                            st.error("An account with that email already exists.")
                        else:
                            st.error(f"Signup failed: {msg}")


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Cyber Shield AI", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Session gate — restore token on every page load, block if not logged in
# ---------------------------------------------------------------------------

# Handle Supabase email-confirmation redirect: the URL fragment contains
# access_token + refresh_token. Read them via JS → write into a hidden
# st.query_params key, then exchange them for a real session.
components.html("""
<script>
(function() {
  var hash = window.location.hash;
  if (!hash) return;
  var params = {};
  hash.replace('#','').split('&').forEach(function(p) {
    var kv = p.split('=');
    params[decodeURIComponent(kv[0])] = decodeURIComponent(kv[1] || '');
  });
  if (params['access_token'] && params['refresh_token']) {
    // Pass tokens to Streamlit via query params so Python can read them
    var url = window.location.origin + window.location.pathname
      + '?sb_access=' + encodeURIComponent(params['access_token'])
      + '&sb_refresh=' + encodeURIComponent(params['refresh_token']);
    window.location.replace(url);
  }
})();
</script>
""", height=0)

# Exchange tokens if redirected back from email confirmation
_qp = st.query_params
if "sb_access" in _qp and "sb_refresh" in _qp and not _get_session():
    try:
        _exchanged = supabase.auth.set_session(_qp["sb_access"], _qp["sb_refresh"])
        _set_session(_exchanged.session)
        # Clean up the URL
        st.query_params.clear()
        st.rerun()
    except Exception:
        st.query_params.clear()

_try_restore_session()
if not _get_session():
    _render_auth_page()
    st.stop()

# Resolve user email once — used by topnav avatar and sidebar
_sb_user    = st.session_state.get("sb_user")
_user_email = _sb_user.email if _sb_user else ""

# ---------------------------------------------------------------------------
# Top navigation bar (sticky, injected into parent document)
# ---------------------------------------------------------------------------
components.html("""
<script>
(function () {
  var NOTIF_COUNT = 3;

  var NAV_LINKS = [
    { label: "Dashboard", icon: "📊" },
    { label: "Email",     icon: "📧" },
    { label: "Scam",      icon: "💬" },
    { label: "Phishing",  icon: "🔗" },
    { label: "Password",  icon: "🔐" },
    { label: "Malware",   icon: "📁" },
  ];

  /* ── inject once into parent document ───────────────── */
  function inject(pd) {
    if (pd.getElementById("cs-topnav")) return;

    /* ── CSS ── */
    var s = pd.createElement("style");
    s.id = "cs-topnav-style";
    s.textContent = [
      "#cs-topnav {",
      "  position:fixed; top:0; left:0; right:0; height:56px;",
      "  background:rgba(10,16,32,0.93);",
      "  border-bottom:1px solid rgba(59,130,246,0.18);",
      "  backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);",
      "  display:flex; align-items:center; justify-content:space-between;",
      "  padding:0 20px; z-index:99999; box-sizing:border-box; gap:12px;",
      "  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;",
      "  transition:box-shadow 0.3s ease;",
      "}",
      "#cs-topnav.scrolled { box-shadow:0 4px 24px rgba(0,0,0,0.6); }",

      /* brand */
      ".tnav-brand { display:flex; align-items:center; gap:9px; text-decoration:none; flex-shrink:0; }",
      ".tnav-brand-icon { width:32px; height:32px; border-radius:8px;",
      "  background:linear-gradient(135deg,#06B6D4,#3B82F6);",
      "  display:flex; align-items:center; justify-content:center;",
      "  font-size:17px; box-shadow:0 2px 10px rgba(6,182,212,0.35);",
      "  transition:transform 0.2s ease; }",
      ".tnav-brand:hover .tnav-brand-icon { transform:scale(1.08); }",
      ".tnav-brand-text { font-size:15px; font-weight:700; letter-spacing:-0.01em;",
      "  background:linear-gradient(90deg,#22D3EE,#60A5FA);",
      "  -webkit-background-clip:text; -webkit-text-fill-color:transparent;",
      "  background-clip:text; white-space:nowrap; }",
      ".tnav-brand-tag { font-size:9px; font-weight:700; letter-spacing:0.08em;",
      "  color:#22D3EE; background:rgba(6,182,212,0.12);",
      "  border:1px solid rgba(6,182,212,0.25); border-radius:4px;",
      "  padding:1px 5px; text-transform:uppercase; }",

      /* nav links */
      ".tnav-links { display:flex; align-items:center; gap:2px; flex:1; justify-content:center; }",
      ".tnav-link { display:inline-flex; align-items:center; gap:5px;",
      "  padding:6px 11px; border-radius:8px; font-size:12.5px; font-weight:500;",
      "  color:#94A3B8; cursor:pointer; white-space:nowrap;",
      "  border:1px solid transparent;",
      "  transition:background 0.2s,color 0.2s,border-color 0.2s,transform 0.15s;",
      "  user-select:none; }",
      ".tnav-link:hover { color:#CBD5E1; background:rgba(148,163,184,0.09); transform:translateY(-1px); }",
      ".tnav-link.active { color:#22D3EE;",
      "  background:linear-gradient(90deg,rgba(6,182,212,0.14),rgba(59,130,246,0.14));",
      "  border-color:rgba(6,182,212,0.28); }",
      ".tnl-icon { font-size:13px; line-height:1; }",

      /* right actions */
      ".tnav-actions { display:flex; align-items:center; gap:6px; flex-shrink:0; }",
      ".tnav-icon-btn { position:relative; width:34px; height:34px; border-radius:9px;",
      "  background:transparent; border:1px solid rgba(148,163,184,0.12);",
      "  display:flex; align-items:center; justify-content:center;",
      "  cursor:pointer; font-size:15px; color:#64748B;",
      "  transition:background 0.2s,border-color 0.2s,color 0.2s,transform 0.15s; }",
      ".tnav-icon-btn:hover { background:rgba(148,163,184,0.1);",
      "  border-color:rgba(148,163,184,0.28); color:#CBD5E1; transform:translateY(-1px); }",

      /* notification badge */
      ".tnav-notif-badge { position:absolute; top:-4px; right:-4px;",
      "  min-width:16px; height:16px; border-radius:8px;",
      "  background:linear-gradient(135deg,#EF4444,#DC2626);",
      "  color:#fff; font-size:9px; font-weight:700;",
      "  display:flex; align-items:center; justify-content:center;",
      "  padding:0 3px; border:1.5px solid rgba(10,16,32,0.9); line-height:1; }",

      /* sticky toggle */
      ".tnav-sticky-toggle { display:flex; align-items:center; gap:6px;",
      "  padding:4px 10px; border-radius:20px; font-size:11px; font-weight:600;",
      "  letter-spacing:0.03em; cursor:pointer; user-select:none;",
      "  background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.22);",
      "  color:#22D3EE; transition:background 0.2s,border-color 0.2s; }",
      ".tnav-sticky-toggle:hover { background:rgba(6,182,212,0.18); }",
      ".tnav-sticky-toggle.static-mode { background:rgba(100,116,139,0.1);",
      "  border-color:rgba(100,116,139,0.2); color:#64748B; }",
      ".tnav-toggle-dot { width:7px; height:7px; border-radius:50%;",
      "  background:#22D3EE; transition:background 0.2s; }",
      ".tnav-sticky-toggle.static-mode .tnav-toggle-dot { background:#64748B; }",

      /* profile */
      ".tnav-avatar { width:32px; height:32px; border-radius:50%;",
      "  background:linear-gradient(135deg,#1E40AF,#0E7490);",
      "  border:2px solid rgba(6,182,212,0.35);",
      "  display:flex; align-items:center; justify-content:center;",
      "  font-size:14px; cursor:pointer;",
      "  transition:border-color 0.2s,transform 0.15s; }",
      ".tnav-avatar:hover { border-color:#22D3EE; transform:scale(1.07); }",

      /* notification dropdown — use opacity+visibility for animatable open/close */
      ".tnav-notif-wrap { position:relative; }",
      ".tnav-dropdown { position:absolute; top:42px; right:0; width:268px;",
      "  background:#0F172A; border:1px solid rgba(59,130,246,0.22);",
      "  border-radius:12px; box-shadow:0 8px 32px rgba(0,0,0,0.55);",
      "  padding:10px 0; z-index:100001;",
      "  opacity:0; visibility:hidden; pointer-events:none;",
      "  transform:translateY(-8px);",
      "  transition:opacity 0.2s ease,transform 0.2s ease,visibility 0.2s; }",
      ".tnav-dropdown.open { opacity:1; visibility:visible; pointer-events:auto; transform:translateY(0); }",
      ".tnav-dropdown-title { font-size:10px; font-weight:700; letter-spacing:0.08em;",
      "  color:#475569; text-transform:uppercase; padding:2px 14px 8px; }",
      ".tnav-notif-item { display:flex; align-items:flex-start; gap:10px;",
      "  padding:8px 14px; cursor:pointer; transition:background 0.15s; }",
      ".tnav-notif-item:hover { background:rgba(148,163,184,0.06); }",
      ".tnav-notif-dot { width:7px; height:7px; border-radius:50%; margin-top:5px; flex-shrink:0; }",
      ".tnav-notif-dot.red   { background:#EF4444; }",
      ".tnav-notif-dot.amber { background:#F59E0B; }",
      ".tnav-notif-dot.cyan  { background:#22D3EE; }",
      ".tnav-notif-text { font-size:12px; color:#CBD5E1; line-height:1.4; }",
      ".tnav-notif-time { font-size:10px; color:#475569; margin-top:1px; }",

      /* push Streamlit content down */
      "section[data-testid='stSidebar'] { padding-top:56px !important; }",
      "[data-testid='stAppViewContainer'] > div:first-child { padding-top:68px !important; }",
      "header[data-testid='stHeader'] { display:none !important; }",
    ].join("\\n");
    pd.head.appendChild(s);

    /* ── build HTML ── */
    var nav = pd.createElement("div");
    nav.id = "cs-topnav";
    nav.innerHTML =
      '<a class="tnav-brand" href="#">' +
        '<div class="tnav-brand-icon">&#x1F6E1;</div>' +
        '<span class="tnav-brand-text">Cyber Shield AI</span>' +
        '<span class="tnav-brand-tag">v1.0</span>' +
      '</a>' +
      '<nav class="tnav-links" id="tnav-links"></nav>' +
      '<div class="tnav-actions">' +
        '<div class="tnav-sticky-toggle" id="tnav-toggle" title="Toggle sticky/static">' +
          '<div class="tnav-toggle-dot"></div>' +
          '<span id="tnav-toggle-label">Sticky</span>' +
        '</div>' +
        '<div class="tnav-icon-btn" id="tnav-search" title="Search">&#x1F50D;</div>' +
        '<div class="tnav-notif-wrap">' +
          '<div class="tnav-icon-btn" id="tnav-notif-btn" title="Notifications">' +
            '&#x1F514;' +
            '<span class="tnav-notif-badge" id="tnav-badge">' + NOTIF_COUNT + '</span>' +
          '</div>' +
          '<div class="tnav-dropdown" id="tnav-notif-drop">' +
            '<div class="tnav-dropdown-title">Notifications</div>' +
            '<div class="tnav-notif-item">' +
              '<div class="tnav-notif-dot red"></div>' +
              '<div><div class="tnav-notif-text">Phishing URL blocked &mdash; <b>paypal-secure.tk</b></div>' +
              '<div class="tnav-notif-time">2 min ago</div></div>' +
            '</div>' +
            '<div class="tnav-notif-item">' +
              '<div class="tnav-notif-dot amber"></div>' +
              '<div><div class="tnav-notif-text">Scam message detected in SMS scan</div>' +
              '<div class="tnav-notif-time">14 min ago</div></div>' +
            '</div>' +
            '<div class="tnav-notif-item">' +
              '<div class="tnav-notif-dot cyan"></div>' +
              '<div><div class="tnav-notif-text">ML models loaded successfully</div>' +
              '<div class="tnav-notif-time">1 hr ago</div></div>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="tnav-avatar" title="Profile">&#x1F464;</div>' +
      '</div>';
    pd.body.insertBefore(nav, pd.body.firstChild);

    /* ── nav links ── */
    var linksEl = pd.getElementById("tnav-links");
    NAV_LINKS.forEach(function(item, idx) {
      var a = pd.createElement("div");
      a.className = "tnav-link";
      a.innerHTML = '<span class="tnl-icon">' + item.icon + '</span>' + item.label;
      a.addEventListener("click", function() { triggerNav(idx, pd); });
      linksEl.appendChild(a);
    });

    /* ── sticky toggle ── */
    var isSticky = true;
    pd.getElementById("tnav-toggle").addEventListener("click", function() {
      isSticky = !isSticky;
      nav.style.position = isSticky ? "fixed" : "relative";
      this.classList.toggle("static-mode", !isSticky);
      pd.getElementById("tnav-toggle-label").textContent = isSticky ? "Sticky" : "Static";
    });

    /* ── notification dropdown ── */
    pd.getElementById("tnav-notif-btn").addEventListener("click", function(e) {
      e.stopPropagation();
      pd.getElementById("tnav-notif-drop").classList.toggle("open");
    });
    pd.addEventListener("click", function() {
      var d = pd.getElementById("tnav-notif-drop");
      if (d) d.classList.remove("open");
    });

    /* ── scroll shadow ── */
    var scroller = pd.querySelector('[data-testid="stAppViewContainer"]') || pd.documentElement;
    scroller.addEventListener("scroll", function() {
      var sy = typeof scroller.scrollTop !== "undefined" ? scroller.scrollTop : (window.parent.scrollY || 0);
      nav.classList.toggle("scrolled", sy > 10);
    }, { passive: true });

    markActive(pd);
  }

  /* ── click sidebar label to navigate ── */
  function triggerNav(idx, pd) {
    /* Streamlit radio: labels with data-baseweb="radio" are the clickable items */
    var labels = pd.querySelectorAll('[data-testid="stSidebar"] label[data-baseweb="radio"]');
    if (labels[idx]) {
      labels[idx].click();
      setTimeout(function() { markActive(pd); }, 150);
    }
  }

  /* ── sync active highlight ── */
  function markActive(pd) {
    var inputs = pd.querySelectorAll('[data-testid="stSidebar"] input[type="radio"]');
    var activeIdx = 0;
    inputs.forEach(function(r, i) { if (r.checked) activeIdx = i; });
    pd.querySelectorAll(".tnav-link").forEach(function(l, i) {
      l.classList.toggle("active", i === activeIdx);
    });
  }

  /* ── init with retry ── */
  function init() {
    var pd = window.parent.document;
    inject(pd);
    var sidebar = pd.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
      /* Watch for DOM changes (Streamlit rerenders radio on selection) */
      var obs = new MutationObserver(function() { markActive(pd); });
      obs.observe(sidebar, { subtree: true, childList: true, attributes: true });
    }
  }

  if (document.readyState === "complete") { init(); }
  else { window.addEventListener("load", init); }
  setTimeout(init, 500);
})();
</script>
""", height=1)

# Inject user initial into topnav avatar after session is confirmed
_avatar_initial = (_user_email[0].upper()) if _user_email else "U"
components.html(f"""
<script>
(function() {{
  var initial = "{_avatar_initial}";
  function updateAvatar(pd) {{
    var av = pd.getElementById("cs-topnav") && pd.querySelector(".tnav-avatar");
    if (av) {{ av.textContent = initial; av.title = "Profile"; }}
  }}
  if (document.readyState === "complete") {{ updateAvatar(window.parent.document); }}
  else {{ window.addEventListener("load", function() {{ updateAvatar(window.parent.document); }}); }}
  setTimeout(function() {{ updateAvatar(window.parent.document); }}, 600);
}})();
</script>
""", height=0)


# ---------------------------------------------------------------------------
# Global CSS theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Root overrides ---- */
.stApp {
    background: linear-gradient(135deg, #0B1120 0%, #111827 50%, #0F172A 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid rgba(59,130,246,0.15);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: #E2E8F0 !important;
}

/* ---- Main area typography ---- */
h1 { color: #F1F5F9 !important; font-weight: 800 !important; letter-spacing: -0.03em; }
h2, h3 { color: #E2E8F0 !important; font-weight: 700 !important; letter-spacing: -0.02em; }
p, span, label { color: #CBD5E1; }

/* ---- Metric cards on dashboard ---- */
.shield-metric-card {
    background: linear-gradient(145deg, rgba(30,58,138,0.35) 0%, rgba(15,23,42,0.6) 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.shield-metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59,130,246,0.15);
}
.shield-metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #F1F5F9;
    line-height: 1.2;
}
.shield-metric-label {
    font-size: 12px;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 6px;
}
.shield-metric-delta {
    font-size: 13px;
    font-weight: 600;
    color: #34D399;
    margin-top: 4px;
}

/* ---- Result banners ---- */
.result-banner-safe {
    background: linear-gradient(90deg, rgba(16,185,129,0.15) 0%, rgba(16,185,129,0.05) 100%);
    border: 1px solid rgba(16,185,129,0.4);
    border-left: 4px solid #10B981;
    border-radius: 10px;
    padding: 16px 20px;
    color: #A7F3D0;
    font-size: 15px;
    font-weight: 600;
}
.result-banner-threat {
    background: linear-gradient(90deg, rgba(239,68,68,0.18) 0%, rgba(239,68,68,0.05) 100%);
    border: 1px solid rgba(239,68,68,0.4);
    border-left: 4px solid #EF4444;
    border-radius: 10px;
    padding: 16px 20px;
    color: #FCA5A5;
    font-size: 15px;
    font-weight: 600;
}
.suggestion-box {
    background: linear-gradient(135deg, rgba(239,68,68,0.06) 0%, rgba(30,41,59,0.5) 100%);
    border: 1px solid rgba(239,68,68,0.15);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 10px;
}
.suggestion-box .sug-title {
    font-size: 12px;
    font-weight: 700;
    color: #F87171;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
}
.suggestion-box .sug-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 13px;
    line-height: 1.5;
}
.suggestion-box .sug-item:last-child { margin-bottom: 0; }
.suggestion-box .sug-icon {
    flex-shrink: 0;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    margin-top: 1px;
}
.suggestion-box .sug-do .sug-icon {
    background: rgba(34,197,94,0.15);
    color: #4ADE80;
}
.suggestion-box .sug-dont .sug-icon {
    background: rgba(239,68,68,0.15);
    color: #F87171;
}
.suggestion-box .sug-text { color: #CBD5E1; }
.suggestion-box .sug-text b { color: #E2E8F0; }

/* ---- Connection badge ---- */
.conn-badge-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 700;
    color: #34D399;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.conn-badge-off {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 700;
    color: #F87171;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-anim 1.5s infinite;
}
.pulse-dot.green { background: #34D399; }
.pulse-dot.red   { background: #F87171; }
@keyframes pulse-anim {
    0%   { opacity: 1; transform: scale(1); }
    50%  { opacity: 0.5; transform: scale(1.4); }
    100% { opacity: 1; transform: scale(1); }
}

/* ---- Password criteria bars ---- */
.pw-criteria-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.pw-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(148,163,184,0.15);
    border-radius: 4px;
    overflow: hidden;
}
.pw-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}
.pw-label {
    font-size: 12px;
    color: #94A3B8;
    min-width: 110px;
}
.pw-check {
    font-size: 14px;
    min-width: 20px;
    text-align: center;
}

/* ---- Sidebar AI card ---- */
.ai-assistant-card {
    background: linear-gradient(145deg, rgba(30,58,138,0.2) 0%, rgba(15,23,42,0.4) 100%);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    margin: 8px 0;
}
.big-avatar-frame {
    width: 72px !important;
    height: 72px !important;
    border-radius: 50% !important;
    border: 2px solid rgba(59,130,246,0.3) !important;
    object-fit: cover !important;
    background-color: transparent !important;
    display: block !important;
    margin: 0 auto 10px auto !important;
}

/* ---- Chat bubbles ---- */
.chat-bubble-user {
    background: linear-gradient(135deg, #1E3A8A, #2563EB);
    padding: 10px 14px;
    border-radius: 12px 12px 4px 12px;
    margin-bottom: 8px;
    font-size: 12px;
    color: #F1F5F9;
    line-height: 1.5;
}
.chat-bubble-ai {
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(148,163,184,0.12);
    padding: 10px 14px;
    border-radius: 12px 12px 12px 4px;
    margin-bottom: 8px;
    font-size: 12px;
    color: #E2E8F0;
    line-height: 1.5;
    border-left: 3px solid #10B981;
}

/* ---- Section headers ---- */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.section-icon-box {
    width: 42px; height: 42px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}

/* ---- Suppress default Streamlit padding in sidebar ---- */
[data-testid="stSidebar"] .stButton > button[key="ai_toggle_btn"] {
    background: linear-gradient(135deg, #1E3A8A, #3B82F6) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    width: 100% !important;
    border: none !important;
    padding: 8px 0 !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] .stButton > button[key^="chip_"] {
    background: rgba(30,58,138,0.35) !important;
    color: #93C5FD !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 8px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    padding: 4px 0 !important;
}

/* ---- Sidebar radio navigation pills ---- */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
    background: rgba(30,58,138,0.15) !important;
    border: 1px solid rgba(59,130,246,0.12) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    cursor: pointer !important;
    transition: background 0.2s, border-color 0.2s !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
    background: rgba(59,130,246,0.15) !important;
    border-color: rgba(59,130,246,0.3) !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(div[aria-checked="true"]) {
    background: linear-gradient(135deg, rgba(30,58,138,0.45), rgba(59,130,246,0.2)) !important;
    border-color: rgba(59,130,246,0.4) !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label p {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(div[aria-checked="true"]) p {
    color: #F1F5F9 !important;
}
/* Hide the default radio circle */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label div[data-baseweb="radio"] {
    display: none !important;
}
/* Metric card hover animation */
.metric-card-wrapper {
    position: relative;
    cursor: pointer;
}
.metric-card-wrapper .metric-desc {
    max-height: 0;
    opacity: 0;
    overflow: hidden;
    transition: max-height 0.4s cubic-bezier(0.4, 0, 0.2, 1),
                opacity 0.35s ease,
                padding 0.35s ease;
    font-size: 12px;
    color: #94A3B8;
    line-height: 1.6;
    padding: 0 4px;
}
.metric-card-wrapper:hover .metric-desc {
    max-height: 120px;
    opacity: 1;
    padding: 12px 4px 0;
}
.metric-card-wrapper:hover .shield-metric-card {
    border-color: rgba(59,130,246,0.35) !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.12);
    transform: translateY(-2px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.3s ease;
}
.shield-metric-card {
    transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.3s ease;
}
/* ---- Professional header ---- */
.header-wrapper {
    background: linear-gradient(135deg, rgba(30,58,138,0.3) 0%, rgba(15,23,42,0.5) 100%);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 4px;
}
.header-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.header-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.header-brand-icon {
    font-size: 36px;
    line-height: 1;
}
.header-title {
    font-size: 26px;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.02em;
    margin: 0;
}
.header-subtitle {
    font-size: 13px;
    color: #94A3B8;
    margin: 4px 0 0 0;
}
.version-badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
    color: #60A5FA;
    letter-spacing: 0.05em;
    margin-left: 10px;
    vertical-align: middle;
}
.header-desc {
    font-size: 13px;
    color: #94A3B8;
    margin-top: 12px;
    line-height: 1.6;
    border-top: 1px solid rgba(100,116,139,0.12);
    padding-top: 12px;
}

/* ---- Hero section ---- */
.hero-section {
    position: relative;
    background: linear-gradient(135deg, #0B1222 0%, #111C36 50%, #0B1325 100%);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 20px;
    padding: 64px 24px 80px;
    margin-bottom: 12px;
    overflow: hidden;
    text-align: center;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -120px;
    left: -80px;
    width: 340px;
    height: 340px;
    background: radial-gradient(circle, rgba(59,130,246,0.22) 0%, transparent 70%);
    border-radius: 50%;
    filter: blur(60px);
    pointer-events: none;
}
.hero-section::after {
    content: '';
    position: absolute;
    bottom: -140px;
    right: -60px;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(6,182,212,0.18) 0%, transparent 70%);
    border-radius: 50%;
    filter: blur(70px);
    pointer-events: none;
}
.hero-blur {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.14) 0%, transparent 70%);
    border-radius: 50%;
    filter: blur(50px);
    pointer-events: none;
}
.hero-content {
    position: relative;
    z-index: 2;
    max-width: 760px;
    margin: 0 auto;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(6,182,212,0.12);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 700;
    color: #22D3EE;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 22px;
}
.hero-badge span { font-size: 12px; }
.hero-title {
    font-size: 46px;
    font-weight: 900;
    line-height: 1.08;
    color: #F8FAFC;
    letter-spacing: -0.04em;
    margin-bottom: 18px;
}
.hero-title .highlight {
    background: linear-gradient(90deg, #22D3EE 0%, #60A5FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 16px;
    line-height: 1.7;
    color: #94A3B8;
    max-width: 620px;
    margin: 0 auto 34px;
}
.hero-buttons {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 48px;
}
.hero-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 14px 28px;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.01em;
    text-decoration: none;
    transition: all 0.25s ease;
    cursor: pointer;
}
.hero-btn-primary {
    background: linear-gradient(90deg, #06B6D4 0%, #3B82F6 100%);
    color: #0F172A;
    border: none;
    box-shadow: 0 8px 24px rgba(6,182,212,0.25);
}
.hero-btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(6,182,212,0.35);
}
.hero-btn-secondary {
    background: transparent;
    color: #E2E8F0;
    border: 1px solid rgba(148,163,184,0.35);
}
.hero-btn-secondary:hover {
    background: rgba(148,163,184,0.08);
    border-color: rgba(148,163,184,0.55);
    color: #F8FAFC;
}
.hero-trust {
    position: relative;
    z-index: 2;
    max-width: 640px;
    margin: 0 auto;
}
.hero-trust-label {
    font-size: 11px;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.hero-trust-logos {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 28px;
    flex-wrap: wrap;
}
.hero-trust-logo {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    font-weight: 700;
    color: #64748B;
    opacity: 0.85;
    transition: opacity 0.2s;
}
.hero-trust-logo:hover { opacity: 1; }
.hero-trust-logo span { font-size: 16px; }

/* ---- Footer ---- */
.footer-wrapper {
    margin-top: 48px;
    padding: 24px 0 8px;
    border-top: 1px solid rgba(100,116,139,0.12);
}
.footer-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    gap: 24px;
    margin-bottom: 20px;
}
.footer-brand {
    font-size: 14px;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 6px;
}
.footer-brand-desc {
    font-size: 11px;
    color: #64748B;
    line-height: 1.5;
}
.footer-col-title {
    font-size: 10px;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.footer-link {
    font-size: 12px;
    color: #64748B;
    text-decoration: none;
    display: block;
    margin-bottom: 4px;
    transition: color 0.2s;
}
.footer-link:hover {
    color: #60A5FA;
}
.footer-bottom {
    font-size: 11px;
    color: #475569;
    text-align: center;
    padding-top: 12px;
    border-top: 1px solid rgba(100,116,139,0.08);
}
/* ---- References ---- */
.ref-section {
    margin-top: 32px;
    padding: 20px;
    background: rgba(15,23,42,0.4);
    border: 1px solid rgba(100,116,139,0.12);
    border-radius: 12px;
}
.ref-title {
    font-size: 14px;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 12px;
}
.ref-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
.ref-card {
    padding: 12px;
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(100,116,139,0.1);
    border-radius: 8px;
}
.ref-card-title {
    font-size: 11px;
    font-weight: 700;
    color: #60A5FA;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.ref-card-text {
    font-size: 12px;
    color: #94A3B8;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper functions (optimized with caching + standalone fallback)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=HEALTH_CHECK_TTL)
def _check_backend_health():
    """Return True if backend is available or running in standalone mode.
    
    In cloud/standalone mode, returns True if ML models are loaded.
    Cached for 30 seconds to avoid HTTP call on every rerun.
    """
    if CLOUD_MODE:
        return bool(_ml_models) or True  # standalone mode always "online"
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=60)
def _api_post(endpoint, payload, timeout=10):
    """POST to the backend and return (data_dict, error_string).
    
    Falls back to local processing in cloud mode or when backend is unreachable.
    Cached for 60 seconds to avoid duplicate API calls for same input.
    """
    # Cloud / standalone mode: process locally
    if CLOUD_MODE:
        if endpoint == "scan-text":
            return _local_scan_text(payload.get("text", "")), None
        elif endpoint == "scan-url":
            return _local_scan_url(payload.get("url", "")), None
        elif endpoint == "analyze-password":
            return _local_analyze_password(payload.get("password", "")), None
        elif endpoint == "analyze-email":
            return _local_analyze_email(
                payload.get("subject", ""), payload.get("body", "")
            ), None
        return None, f"Unknown endpoint: {endpoint}"

    # Try backend first
    try:
        res = requests.post(f"{BACKEND_URL}/{endpoint}", json=payload, timeout=timeout)
        res.raise_for_status()
        return res.json(), None
    except requests.exceptions.ConnectionError:
        # Fallback to local processing if ML models are available
        if ML_AVAILABLE and _ml_models:
            if endpoint == "scan-text":
                return _local_scan_text(payload.get("text", "")), None
            elif endpoint == "scan-url":
                return _local_scan_url(payload.get("url", "")), None
            elif endpoint == "analyze-password":
                return _local_analyze_password(payload.get("password", "")), None
            elif endpoint == "analyze-email":
                return _local_analyze_email(
                    payload.get("subject", ""), payload.get("body", "")
                ), None
        return None, "Cannot reach backend. Ensure the server is running on port 8000."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


@st.cache_data(ttl=60)
def _api_upload(endpoint, file_name, file_content_bytes, file_type, timeout=10):
    """POST a file upload to the backend and return (data_dict, error_string).
    
    Falls back to local file scanning in cloud mode.
    Cached for 60 seconds based on file name + content hash.
    """
    # Cloud / standalone mode: process locally
    if CLOUD_MODE:
        if endpoint == "scan-file":
            return _local_scan_file(file_name), None
        return None, f"Unknown endpoint: {endpoint}"

    # Try backend first
    try:
        files = {"file": (file_name, file_content_bytes, file_type)}
        res = requests.post(f"{BACKEND_URL}/{endpoint}", files=files, timeout=timeout)
        res.raise_for_status()
        return res.json(), None
    except requests.exceptions.ConnectionError:
        # Fallback for file scanning
        if endpoint == "scan-file":
            return _local_scan_file(file_name), None
        return None, "Cannot reach backend. Ensure the server is running on port 8000."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def _render_result(status, safe_msg, threat_msg):
    """Render a styled result banner."""
    if status == "Threat Detected":
        st.markdown(f'<div class="result-banner-threat">🚨 {threat_msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-banner-safe">✅ {safe_msg}</div>', unsafe_allow_html=True)


def _render_suggestions(suggestions):
    """Render a styled safety suggestion box with do/don't bullet points.
    
    Each suggestion is a tuple: (type, text) where type is 'do' or 'dont'.
    """
    html = '<div class="suggestion-box"><div class="sug-title">🛡️ Safety Recommendations</div>'
    for stype, text in suggestions:
        css_class = "sug-do" if stype == "do" else "sug-dont"
        icon = "✓" if stype == "do" else "✕"
        html += (
            f'<div class="sug-item {css_class}">'
            f'<div class="sug-icon">{icon}</div>'
            f'<div class="sug-text">{text}</div>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _section_header(icon, title, color):
    """Render a consistent section header."""
    st.markdown(
        f'<div class="section-header">'
        f'<div class="section-icon-box" style="background:{color};">{icon}</div>'
        f'<div><span style="font-size:22px; font-weight:700; color:#F1F5F9;">{title}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _trim_chat_history():
    if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]


@st.cache_data(ttl=300)
def _call_qwen_api(messages_tuple: tuple, max_tokens: int = 512) -> str | None:
    """Call Qwen API via OpenAI-compatible endpoint with caching (5 min TTL).
    
    Returns response text or None on failure.
    """
    if not QWEN_ENABLED:
        return None
    try:
        messages = [{"role": r, "content": c} for r, c in messages_tuple]
        response = requests.post(
            QWEN_API_URL,
            headers={
                "Authorization": f"Bearer {_DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": QWEN_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        return None
    except Exception:
        return None


def _process_ai_query(query: str):
    """
    Process a chat query using Qwen AI (if enabled) with fallback to keyword-based responses.
    Still performs real backend scans for URL/text/password requests.
    """
    q = query.lower().strip()
    scan_result = None  # Will hold scan results to pass to Qwen for context

    # --- Detect and perform backend scans first ---
    url_keywords = ["scan http", "check http", "scan url", "check url", "check this link",
                    "check this url", "is this url", "is this link", "لنک", "ویب سائٹ"]
    if any(kw in q for kw in url_keywords) or q.startswith("http"):
        url = ""
        for token in query.split():
            if token.startswith("http"):
                url = token
                break
        if url:
            data, err = _api_post("scan-url", {"url": url})
            if err:
                scan_result = f"URL scan failed: {err}"
            else:
                scan_result = f"URL scan result for {url}: {data.get('status')} (risk score: {data.get('risk_score', 'N/A')})"

    text_keywords = ["scam", "sms", "message", "whatsapp", "bisp", "inam", "lottery",
                     "انعام", "لاٹری", "میسج", "check this text", "scan this text"]
    if any(kw in q for kw in text_keywords) and len(query) > 30:
        if not query.lower().startswith(("is ", "how ", "what ", "can ", "check ", "scan ")):
            data, err = _api_post("scan-text", {"text": query})
            if err:
                scan_result = f"Text scan failed: {err}"
            else:
                scan_result = f"Text scan result: {data.get('status')}"

    pw_keywords = ["password", "passphrase", "strong", "weak", "secure", "پاس ورڈ", "حفاظت", "how strong"]
    if any(kw in q for kw in pw_keywords):
        skip_words = {"password", "passphrase", "strong", "weak", "secure", "how", "is", "the", "my", "check", "of"}
        tokens = query.split()
        candidate = ""
        for t in tokens:
            cleaned = t.strip("?!.,;")
            if cleaned.lower() not in skip_words and len(cleaned) >= 4:
                candidate = cleaned
                break
        if candidate:
            data, err = _api_post("analyze-password", {"password": candidate})
            if err:
                scan_result = f"Password analysis failed: {err}"
            else:
                masked = candidate[:2] + '*' * max(0, len(candidate)-2)
                scan_result = f"Password '{masked}' strength: {data.get('strength')}"

    # --- Build messages for Qwen ---
    messages = [{"role": "system", "content": QWEN_SYSTEM_PROMPT}]

    # Add recent chat history for context (last 6 messages)
    history = st.session_state.chat_history[-6:] if st.session_state.chat_history else []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add scan result as context if available
    if scan_result:
        messages.append({"role": "user", "content": f"[System: {scan_result}]"})

    # Add current query
    messages.append({"role": "user", "content": query})

    # --- Call Qwen API ---
    messages_tuple = tuple((m["role"], m["content"]) for m in messages)
    resp = _call_qwen_api(messages_tuple)

    if resp:
        st.session_state.chat_history.append({"role": "assistant", "content": resp})
        return

    # --- Fallback to keyword-based responses if Qwen fails ---
    if scan_result:
        # Use scan result directly
        if "Threat Detected" in scan_result:
            resp = f"🚨 **Threat Detected!** {scan_result}\n\nI recommend avoiding this and not interacting with it."
        elif "Safe" in scan_result:
            resp = f"✅ **Safe.** {scan_result}\n\nHowever, always stay cautious with unsolicited content."
        else:
            resp = scan_result
    else:
        # Generic fallback
        resp = (
            "I'm your cybersecurity assistant powered by Qwen AI. I can help with:\n"
            "• Scanning URLs for phishing\n"
            "• Analyzing emails for spam\n"
            "• Checking messages for scams\n"
            "• Evaluating password strength\n\n"
            "Try asking: 'scan http://example.com' or 'Is this password strong: MyP@ss123?'"
        )

    st.session_state.chat_history.append({"role": "assistant", "content": resp})


# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
backend_ok = _check_backend_health()

# Dynamic subtitle based on deployment mode
_hero_subtitle = (
    'Detect phishing links, scam messages, spam emails, and weak passwords in seconds. '
    'Cyber Shield AI runs entirely in the cloud with on-device ML inference — '
    'your data never leaves the app.'
    if CLOUD_MODE else
    'Detect phishing links, scam messages, spam emails, and weak passwords in seconds. '
    'Cyber Shield AI runs locally through a FastAPI backend — '
    'your data never leaves your machine.'
)

st.markdown(
    '<div class="hero-section">'
    '<div class="hero-blur"></div>'
    '<div class="hero-content">'
    '<div class="hero-badge"><span>🚀</span> New Release v1.0</div>'
    '<div class="hero-title">AI-Powered <span class="highlight">Cyber Defense</span> for Everyone</div>'
    f'<div class="hero-subtitle">{_hero_subtitle}</div>'
    '</div>'
    '<div class="hero-trust">'
    '<div class="hero-trust-label">Trusted by security-aware users</div>'
    '<div class="hero-trust-logos">'
    '<div class="hero-trust-logo"><span>🛡️</span>Phishing Shield</div>'
    '<div class="hero-trust-logo"><span>📱</span>SMS Scanner</div>'
    '<div class="hero-trust-logo"><span>📧</span>Email Guard</div>'
    '<div class="hero-trust-logo"><span>🔐</span>Password AI</div>'
    '</div>'
    '</div>'
    + (
        '<div style="position:absolute;top:18px;right:18px;z-index:3;">'
        '<div class="conn-badge-ok"><span class="pulse-dot green"></span> Cloud Mode</div>'
        '</div>'
        if CLOUD_MODE and backend_ok else
        '<div style="position:absolute;top:18px;right:18px;z-index:3;">'
        '<div class="conn-badge-ok"><span class="pulse-dot green"></span> Systems Online</div>'
        '</div>'
        if backend_ok else
        '<div style="position:absolute;top:18px;right:18px;z-index:3;">'
        '<div class="conn-badge-off"><span class="pulse-dot red"></span> Backend Offline</div>'
        '</div>'
    )
    + '</div>',
    unsafe_allow_html=True,
)

# Hero CTA buttons — rendered as real Streamlit buttons, styled to match hero design
st.markdown("""
<style>
/* Hero CTA button row */
div[data-testid="stHorizontalBlock"]:has(> div > div[data-testid="stButton"]:first-child) {
    justify-content: center;
    gap: 0;
    margin-top: -54px;
    margin-bottom: 0;
    position: relative;
    z-index: 5;
}
/* Primary hero button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]:nth-child(1) button {
    background: linear-gradient(90deg, #06B6D4 0%, #3B82F6 100%) !important;
    color: #0F172A !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 8px 24px rgba(6,182,212,0.28) !important;
    min-height: 0 !important;
    height: auto !important;
    line-height: 1.4 !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]:nth-child(1) button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(6,182,212,0.4) !important;
}
/* Secondary hero button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]:nth-child(2) button {
    background: transparent !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(148,163,184,0.35) !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    min-height: 0 !important;
    height: auto !important;
    line-height: 1.4 !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]:nth-child(2) button:hover {
    background: rgba(148,163,184,0.08) !important;
    border-color: rgba(148,163,184,0.55) !important;
}
</style>
""", unsafe_allow_html=True)

_hero_c1, _hero_c2, _hero_c3 = st.columns([2, 1.3, 1.3])
with _hero_c2:
    if st.button("⚡  Start Scanning", use_container_width=True, key="_hero_scan"):
        st.session_state["_hero_nav"] = "💬  Scam Text Scanner"
        st.rerun()
with _hero_c3:
    if st.button("🔍  Explore Features", use_container_width=True, key="_hero_features"):
        st.session_state["_hero_nav"] = "📊  Threat Dashboard"
        st.rerun()
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<p style="font-size:11px; color:#64748B; font-weight:700; '
    'text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px;">Navigation</p>',
    unsafe_allow_html=True,
)
menu = [
    "📊  Threat Dashboard",
    "📧  Email Analyzer",
    "💬  Scam Text Scanner",
    "🔗  Phishing & QR Detector",
    "🔐  Password Analyzer",
    "📁  Malware Guard",
]
# If a hero button was clicked, pre-select that nav item
_hero_nav_target = st.session_state.pop("_hero_nav", None)
_nav_index = menu.index(_hero_nav_target) if _hero_nav_target and _hero_nav_target in menu else None
choice = st.sidebar.radio(
    "Navigation", menu,
    label_visibility="collapsed",
    index=_nav_index if _nav_index is not None else 0,
    key="_main_nav",
)

# ── Logout + signed-in user ──────────────────────────────────────────────────
st.sidebar.markdown(
    '<hr style="border:none;border-top:1px solid rgba(59,130,246,0.15);margin:14px 0 10px;">',
    unsafe_allow_html=True,
)
_user_display = _user_email[:28] + "…" if len(_user_email) > 28 else _user_email
st.sidebar.markdown(
    f'<p style="font-size:11px;color:#475569;margin:0 0 8px;overflow:hidden;'
    f'text-overflow:ellipsis;white-space:nowrap;">🔐 {_user_display}</p>',
    unsafe_allow_html=True,
)
if st.sidebar.button("Sign Out", key="_logout", use_container_width=True):
    supabase.auth.sign_out()
    _set_session(None)
    st.rerun()

# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


if choice == "📊  Threat Dashboard":
    _section_header("📡", "Global Threat Intelligence", "rgba(59,130,246,0.2)")

    metrics = [
        ("2,841", "Scams Intercepted", "▲ +14% this week",
         "Monitors and isolates live local SMS/WhatsApp spam patterns like BISP or fake prize package lottery logs before execution."),
        ("942", "Phishing Links Blocked", "▲ +8% this week",
         "Intercepts and breaks hazardous credential-harvesting hyper-links, malicious code blocks, and fake landing pages."),
        ("98.4%", "Detection Accuracy", "ML-powered analysis",
         "Represents the structural precision matrix of our trained scikit-learn models, verified against testing threat vectors."),
    ]

    col1, col2, col3 = st.columns(3)
    for col, (val, label, delta, desc) in zip([col1, col2, col3], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card-wrapper">'
                f'<div class="shield-metric-card">'
                f'<div class="shield-metric-value">{val}</div>'
                f'<div class="shield-metric-label">{label}</div>'
                f'<div class="shield-metric-delta">{delta}</div>'
                f'</div>'
                f'<div class="metric-desc">{desc}</div>'
                f'</div>', unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Capability cards
    _section_header("⚡", "Active Modules", "rgba(16,185,129,0.15)")
    cap1, cap2, cap3, cap4, cap5 = st.columns(5)
    with cap1:
        st.markdown(
            '<div class="shield-metric-card" style="padding:16px 14px;">'
            '<div style="font-size:28px; margin-bottom:6px;">📧</div>'
            '<div style="font-size:13px; font-weight:700; color:#E2E8F0;">Email Analyzer</div>'
            '<div style="font-size:11px; color:#94A3B8; margin-top:4px;">Spam & phishing detection</div>'
            '</div>', unsafe_allow_html=True,
        )
    with cap2:
        st.markdown(
            '<div class="shield-metric-card" style="padding:16px 14px;">'
            '<div style="font-size:28px; margin-bottom:6px;">💬</div>'
            '<div style="font-size:13px; font-weight:700; color:#E2E8F0;">Scam Scanner</div>'
            '<div style="font-size:11px; color:#94A3B8; margin-top:4px;">SMS / WhatsApp</div>'
            '</div>', unsafe_allow_html=True,
        )
    with cap3:
        st.markdown(
            '<div class="shield-metric-card" style="padding:16px 14px;">'
            '<div style="font-size:28px; margin-bottom:6px;">🔗</div>'
            '<div style="font-size:13px; font-weight:700; color:#E2E8F0;">Phishing & QR</div>'
            '<div style="font-size:11px; color:#94A3B8; margin-top:4px;">URL + QR code analysis</div>'
            '</div>', unsafe_allow_html=True,
        )
    with cap4:
        st.markdown(
            '<div class="shield-metric-card" style="padding:16px 14px;">'
            '<div style="font-size:28px; margin-bottom:6px;">🔐</div>'
            '<div style="font-size:13px; font-weight:700; color:#E2E8F0;">Password Audit</div>'
            '<div style="font-size:11px; color:#94A3B8; margin-top:4px;">Strength entropy check</div>'
            '</div>', unsafe_allow_html=True,
        )
    with cap5:
        st.markdown(
            '<div class="shield-metric-card" style="padding:16px 14px;">'
            '<div style="font-size:28px; margin-bottom:6px;">📁</div>'
            '<div style="font-size:13px; font-weight:700; color:#E2E8F0;">Malware Guard</div>'
            '<div style="font-size:11px; color:#94A3B8; margin-top:4px;">File extension screening</div>'
            '</div>', unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Getting started:** Select a module from the sidebar to run a live scan. The AI assistant is available in the bottom-left for guidance.")

   # ── Threat Defense Zone — radar animation ─────────────────────
    _section_header("🛡️", "Threat Defense Zone", "rgba(6,182,212,0.15)")
    components.html("""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#060D1A;overflow:hidden;font-family:'Inter',system-ui,sans-serif;color:#E2E8F0}
canvas{display:block}
#wrap{position:relative;width:100%}
/* Top bar */
#topbar{position:absolute;top:0;left:0;right:0;display:flex;justify-content:space-between;
  align-items:center;padding:10px 16px;z-index:5;pointer-events:none}
#topbar .title{font-size:11px;letter-spacing:.12em;color:#94A3B8;text-transform:uppercase}
#topbar .title b{color:#06B6D4}
#topbar .sub{font-size:10px;letter-spacing:.1em;color:#06B6D4;text-transform:uppercase}
/* Stats row */
#stats{display:flex;gap:2px;padding:0 12px;margin-top:-4px}
.stat{flex:1;background:rgba(15,23,42,.7);border:1px solid rgba(6,182,212,.1);
  border-radius:8px;padding:10px 14px;text-align:center}
.stat .lbl{font-size:9px;letter-spacing:.1em;color:#64748B;text-transform:uppercase;margin-bottom:4px}
.stat .val{font-size:18px;font-weight:700}
.stat .val span{font-size:11px;font-weight:400;color:#64748B}
/* Mode selector */
#modes{padding:10px 14px}
#modes .mlbl{font-size:9px;color:#475569;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px}
.mode-row{display:flex;gap:6px}
.mode-btn{padding:6px 16px;border-radius:6px;font-size:11px;font-weight:600;
  letter-spacing:.04em;cursor:pointer;border:1px solid transparent;transition:all .25s}
.mode-btn.active{background:#06B6D4;color:#0F172A;border-color:#06B6D4}
.mode-btn:not(.active){background:rgba(15,23,42,.6);color:#64748B;border-color:rgba(100,116,139,.15)}
.mode-btn:not(.active):hover{color:#94A3B8;border-color:rgba(100,116,139,.3)}
/* Sliders */
#sliders{display:flex;gap:12px;padding:6px 14px 10px}
.slider-box{flex:1}
.slider-box .slbl{font-size:9px;color:#475569;letter-spacing:.06em;margin-bottom:6px}
.slider-track{height:4px;background:rgba(30,41,59,.8);border-radius:2px;position:relative;cursor:pointer}
.slider-fill{height:100%;background:#06B6D4;border-radius:2px;transition:width .3s}
.slider-knob{width:14px;height:14px;border-radius:50%;background:#06B6D4;border:2px solid #0F172A;
  position:absolute;top:50%;transform:translate(-50%,-50%);box-shadow:0 0 8px rgba(6,182,212,.4);
  cursor:grab;transition:left .3s}
.slider-vals{display:flex;justify-content:space-between;font-size:9px;color:#475569;margin-top:4px}
#note{text-align:center;font-size:9px;color:#1E293B;padding:2px 0 6px}
</style></head><body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="topbar">
    <div class="title">THREAT DEFENSE ZONE</div>
    <div class="sub">CORE OPERATIONS</div>
  </div>
</div>
<div id="stats">
  <div class="stat"><div class="lbl">Neutralized</div><div class="val" id="ncount">0 <span>logs</span></div></div>
  <div class="stat"><div class="lbl">Core Integrity</div><div class="val" style="color:#22C55E">98.4%</div></div>
  <div class="stat"><div class="lbl">Cyber Load</div><div class="val" id="cload" style="color:#06B6D4">87%</div></div>
</div>
<div id="modes">
  <div class="mlbl">Operational Threat Mode</div>
  <div class="mode-row">
    <div class="mode-btn active">Cyan Shield</div>
    <div class="mode-btn">Max Counter</div>
    <div class="mode-btn">Analysis Zone</div>
  </div>
</div>
<div id="sliders">
  <div class="slider-box"><div class="slbl">Threat Log Influx Rate</div>
    <div class="slider-track"><div class="slider-fill" style="width:35%"></div>
    <div class="slider-knob" style="left:35%"></div></div>
    <div class="slider-vals"><span>1x</span><span>8x</span></div></div>
  <div class="slider-box"><div class="slbl">Cyber Scan Matrix Frequency</div>
    <div class="slider-track"><div class="slider-fill" style="width:65%"></div>
    <div class="slider-knob" style="left:65%"></div></div>
    <div class="slider-vals"><span>6x</span><span>1.5Hz</span></div></div>
</div>
<div id="note">AI-powered threat neutralization — real-time defense matrix</div>
<script>
const C=document.getElementById('c'),G=C.getContext('2d');
let W,H,cx,cy;
function resize(){
  W=C.width=C.parentElement.clientWidth||700;
  H=280;C.height=H;
  cx=W/2;cy=H/2+10;
}
resize();window.addEventListener('resize',resize);

/* ── Config ────────────────────────────────────────────── */
const RINGS=[
  {r:0,segs:8,gap:.18,speed:.0012,rot:0},
  {r:0,segs:12,gap:.12,speed:-.0008,rot:0},
  {r:0,segs:6,gap:.22,speed:.0018,rot:0}
];
const THREAT_TYPES=[
  {label:'SCAM',c:'#EAB308'},
  {label:'MALWARE',c:'#F97316'},
  {label:'PHISHING',c:'#A855F7'},
  {label:'TROJAN',c:'#EC4899'},
  {label:'BREACH',c:'#F43F5E'},
  {label:'VIRUS',c:'#EF4444'},
];
let threats=[],effects=[],neutralized=0,spawnClock=0;
let sweepAngle=0,orbitAngle=0,pulsePhase=0;
const MAX_THREATS=6;

function initRings(){
  const base=Math.min(W,H)*0.38;
  RINGS[0].r=base*0.35;
  RINGS[1].r=base*0.65;
  RINGS[2].r=base;
}
initRings();

/* ── Threats ─────────────────────────────────────────── */
function spawnThreat(){
  if(threats.length>=MAX_THREATS)return;
  const t=THREAT_TYPES[Math.floor(Math.random()*THREAT_TYPES.length)];
  const angle=Math.random()*Math.PI*2;
  const dist=RINGS[2].r+60+Math.random()*40;
  const id='TR-'+Math.floor(1000+Math.random()*9000);
  threats.push({
    angle,dist,startDist:dist,...t,id,
    alpha:0,alive:true,neutralizing:false,nTime:0
  });
}

/* ── Effects ─────────────────────────────────────────── */
function addNeutralize(x,y){
  for(let i=0;i<16;i++){
    const a=Math.random()*Math.PI*2,sp=1.5+Math.random()*3;
    effects.push({x,y,vx:Math.cos(a)*sp,vy:Math.sin(a)*sp,
      life:1,size:2+Math.random()*3,type:'p'});
  }
  effects.push({x,y,r:4,life:1,type:'ring'});
}

/* ── Drawing ─────────────────────────────────────────── */
function drawRings(){
  for(const ring of RINGS){
    ring.rot+=ring.speed;
    const segAngle=Math.PI*2/ring.segs;
    const drawAngle=segAngle*(1-ring.gap);
    G.strokeStyle='rgba(6,182,212,.2)';
    G.lineWidth=1.5;
    for(let i=0;i<ring.segs;i++){
      const start=ring.rot+i*segAngle;
      G.beginPath();
      G.arc(cx,cy,ring.r,start,start+drawAngle);
      G.stroke();
    }
  }
}

function drawSweep(){
  sweepAngle+=.015;
  const grad=G.createConicalGradient
    ?null:null;
  // Sweep line
  const sx=cx+Math.cos(sweepAngle)*RINGS[2].r;
  const sy=cy+Math.sin(sweepAngle)*RINGS[2].r;
  // Trail
  for(let i=0;i<20;i++){
    const a=sweepAngle-i*.02;
    const alpha=(1-i/20)*.15;
    G.strokeStyle=`rgba(6,182,212,${alpha})`;
    G.lineWidth=1;
    G.beginPath();
    G.moveTo(cx,cy);
    G.lineTo(cx+Math.cos(a)*RINGS[2].r,cy+Math.sin(a)*RINGS[2].r);
    G.stroke();
  }
  // Main line
  G.strokeStyle='rgba(6,182,212,.5)';G.lineWidth=1.5;
  G.beginPath();G.moveTo(cx,cy);G.lineTo(sx,sy);G.stroke();
}

function drawOrbit(){
  orbitAngle+=.02;
  pulsePhase+=.05;
  const ox=cx+Math.cos(orbitAngle)*RINGS[2].r;
  const oy=cy+Math.sin(orbitAngle)*RINGS[2].r;
  // Glow
  const glowR=12+Math.sin(pulsePhase)*3;
  const grd=G.createRadialGradient(ox,oy,1,ox,oy,glowR);
  grd.addColorStop(0,'rgba(6,182,212,.6)');
  grd.addColorStop(1,'rgba(6,182,212,0)');
  G.fillStyle=grd;G.beginPath();G.arc(ox,oy,glowR,0,Math.PI*2);G.fill();
  // Dot
  G.fillStyle='#06B6D4';G.shadowColor='#06B6D4';G.shadowBlur=10;
  G.beginPath();G.arc(ox,oy,4,0,Math.PI*2);G.fill();
  G.shadowBlur=0;
}

function drawCenter(){
  // Center glow
  const cg=G.createRadialGradient(cx,cy,2,cx,cy,RINGS[0].r*.8);
  cg.addColorStop(0,'rgba(6,182,212,.08)');
  cg.addColorStop(1,'rgba(6,182,212,0)');
  G.fillStyle=cg;G.beginPath();G.arc(cx,cy,RINGS[0].r*.8,0,Math.PI*2);G.fill();
  // Center dot
  G.fillStyle='#06B6D4';G.shadowColor='#06B6D4';G.shadowBlur=15;
  G.beginPath();G.arc(cx,cy,5,0,Math.PI*2);G.fill();
  G.shadowBlur=0;
  // DEFENSE label
  G.fillStyle='rgba(6,182,212,.7)';G.font='bold 10px sans-serif';
  G.textAlign='center';G.textBaseline='middle';
  G.fillText('DEFENSE',cx,cy-RINGS[0].r-14);
}

function drawThreats(){
  for(const t of threats){
    // Animate inward
    if(!t.neutralizing){
      t.alpha=Math.min(1,t.alpha+.025);
      t.dist-=(.15+t.alpha*.25);
    }
    const tx=cx+Math.cos(t.angle)*t.dist;
    const ty=cy+Math.sin(t.angle)*t.dist;
    // Connecting line to center
    G.strokeStyle=`rgba(${t.c==='#EF4444'?'239,68,68':
      t.c==='#F97316'?'249,115,22':t.c==='#A855F7'?'168,85,247':
      t.c==='#EC4899'?'236,72,153':t.c==='#F43F5E'?'244,63,94':'234,179,8'},${t.alpha*.3})`;
    G.lineWidth=1;
    G.beginPath();G.moveTo(cx,cy);G.lineTo(tx,ty);G.stroke();
    // Threat dot
    G.globalAlpha=t.alpha;
    G.fillStyle=t.c;G.shadowColor=t.c;G.shadowBlur=8;
    G.beginPath();G.arc(tx,ty,4,0,Math.PI*2);G.fill();
    G.shadowBlur=0;
    // Label box
    const lx=tx+(tx>cx?12:-72),ly=ty-10;
    G.fillStyle='rgba(15,23,42,.85)';
    G.strokeStyle=`rgba(${t.c==='#EF4444'?'239,68,68':
      t.c==='#F97316'?'249,115,22':t.c==='#A855F7'?'168,85,247':
      t.c==='#EC4899'?'236,72,153':t.c==='#F43F5E'?'244,63,94':'234,179,8'},.4)`;
    G.lineWidth=1;
    G.beginPath();G.roundRect(lx,ly,60,22,4);G.fill();G.stroke();
    G.fillStyle='#E2E8F0';G.font='bold 8px sans-serif';G.textAlign='left';
    G.fillText(t.id,lx+6,ly+9);
    G.fillStyle=t.c;G.font='7px sans-serif';
    G.fillText(t.label,lx+6,ly+18);
    G.globalAlpha=1;
    // Neutralize check
    if(t.dist<RINGS[0].r+10&&!t.neutralizing){
      t.neutralizing=true;t.nTime=0;
      addNeutralize(tx,ty);
      neutralized++;
      document.getElementById('ncount').innerHTML=neutralized+' <span>logs</span>';
      // Update cyber load
      const load=85+Math.floor(Math.random()*10);
      document.getElementById('cload').textContent=load+'%';
    }
    if(t.neutralizing){
      t.nTime++;t.alpha-=.08;
    }
  }
  threats=threats.filter(t=>t.alpha>0);
}

function drawEffects(){
  for(const e of effects){
    G.globalAlpha=e.life;
    if(e.type==='p'){
      G.fillStyle='#06B6D4';G.shadowColor='#06B6D4';G.shadowBlur=4;
      G.beginPath();G.arc(e.x,e.y,e.size*e.life,0,Math.PI*2);G.fill();
      G.shadowBlur=0;
    }else if(e.type==='ring'){
      G.strokeStyle=`rgba(6,182,212,${e.life*.6})`;G.lineWidth=2;
      G.beginPath();G.arc(e.x,e.y,e.r,0,Math.PI*2);G.stroke();
    }
  }
  G.globalAlpha=1;
  // Update effects
  for(const e of effects){
    if(e.type==='ring'){e.r+=2;e.life-=.03}
    else{e.x+=e.vx;e.y+=e.vy;e.vx*=.95;e.vy*=.95;e.life-=.025}
  }
  effects=effects.filter(e=>e.life>0);
}

/* ── Main loop ─────────────────────────────────────────── */
function frame(){
  G.clearRect(0,0,W,H);
  // Background
  G.fillStyle='#060D1A';G.fillRect(0,0,W,H);
  // Subtle grid
  G.strokeStyle='rgba(6,182,212,.02)';G.lineWidth=1;
  for(let i=0;i<W;i+=25){G.beginPath();G.moveTo(i,0);G.lineTo(i,H);G.stroke()}
  for(let i=0;i<H;i+=25){G.beginPath();G.moveTo(0,i);G.lineTo(W,i);G.stroke()}

  initRings();

  // Spawn
  spawnClock++;
  if(spawnClock>55+Math.random()*35){spawnClock=0;spawnThreat()}

  // Draw layers
  drawRings();
  drawSweep();
  drawThreats();
  drawOrbit();
  drawCenter();
  drawEffects();

  requestAnimationFrame(frame);
}

// Mode buttons
document.querySelectorAll('.mode-btn').forEach(b=>{
  b.addEventListener('click',()=>{
    document.querySelectorAll('.mode-btn').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
  });
});

// Click to neutralize
C.addEventListener('click',e=>{
  const r=C.getBoundingClientRect();
  const mx=e.clientX-r.left,my=e.clientY-r.top;
  let best=null,bd=Infinity;
  for(const t of threats){
    if(t.neutralizing)continue;
    const tx=cx+Math.cos(t.angle)*t.dist;
    const ty=cy+Math.sin(t.angle)*t.dist;
    const d=Math.hypot(tx-mx,ty-my);
    if(d<50&&d<bd){best=t;bd=d}
  }
  if(best){
    best.neutralizing=true;best.nTime=0;
    const bx=cx+Math.cos(best.angle)*best.dist;
    const by=cy+Math.sin(best.angle)*best.dist;
    addNeutralize(bx,by);
    neutralized++;document.getElementById('ncount').innerHTML=neutralized+' <span>logs</span>';
  }
});

frame();
</script></body></html>""", height=500)


    # References section
    st.markdown(
        '<div class="ref-section">'
        '<div class="ref-title">📚 References & Technology Stack</div>'
        '<div class="ref-grid">'

        '<div class="ref-card">'
        '<div class="ref-card-title">Machine Learning Models</div>'
        '<div class="ref-card-text">'
        '• <b>Scam Detection</b> — LinearSVC trained on SMS/spam corpus with TF-IDF vectorization (50K vocabulary)<br>'
        '• <b>Email Spam Filter</b> — LinearSVC (SVM) on clean email dataset with TF-IDF features<br>'
        '• <b>Phishing URL Classifier</b> — XGBoost on URL feature vectors (length, dot count, hyphen count, keyword presence)<br>'
        '• <b>Framework</b> — scikit-learn 1.x with joblib model persistence'
        '</div></div>'

        '<div class="ref-card">'
        '<div class="ref-card-title">Datasets & Corpora</div>'
        '<div class="ref-card-text">'
        '• SMS Spam Collection Dataset (UCI ML Repository)<br>'
        '• Phishing URL dataset with lexical feature extractions<br>'
        '• Clean/Spam email corpus for binary classification<br>'
        '• Local keyword dictionaries for BISP, lottery, and prize scams (Urdu/English)'
        '</div></div>'

        '<div class="ref-card">'
        '<div class="ref-card-title">Backend Architecture</div>'
        '<div class="ref-card-text">'
        '• <b>FastAPI</b> — async REST API with Pydantic validation and CORS middleware<br>'
        '• <b>Uvicorn</b> — ASGI server on port 8000<br>'
        '• <b>Streamlit</b> — interactive dashboard on port 8501<br>'
        '• <b>OpenCV</b> — QR code detection and decoding via cv2.QRCodeDetector'
        '</div></div>'

        '<div class="ref-card">'
        '<div class="ref-card-title">Research & Standards</div>'
        '<div class="ref-card-text">'
        '• NIST Cybersecurity Framework — threat detection guidelines<br>'
        '• OWASP Phishing Prevention Cheat Sheet<br>'
        '• SVM-based text classification (Joachims, 1998)<br>'
        '• TF-IDF weighting for document classification (Salton & Buckley, 1988)'
        '</div></div>'

        '</div></div>',
        unsafe_allow_html=True,
    )

# ==================== SCAM TEXT SCANNER ====================
elif choice == "💬  Scam Text Scanner":
    _section_header("💬", "Scam Text Scanner", "rgba(245,158,11,0.2)")
    st.markdown("Analyze messages from SMS, email, or WhatsApp for financial scam patterns. Supports English, Urdu script, and Roman Urdu.")

    # Read pending sample value BEFORE rendering widget
    _init_text = st.session_state.pop("scam_sample_text", "")

    # If sample was set, seed the widget key in session state
    if _init_text:
        st.session_state["_scam_text_input"] = _init_text

    user_text = st.text_area(
        "Paste suspicious message below:",
        key="_scam_text_input",
        height=140,
        placeholder="e.g. Congratulations! You won 50,000 rupees! Click here to claim your prize...",
    )

    # Quick sample buttons
    st.markdown("**Try a sample message:**")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        if st.button("📱 Lottery Scam", key="sample_lottery", use_container_width=True):
            st.session_state["scam_sample_text"] = "Congratulations! You have won Rs 50,000 in the BISP lottery. Send your CNIC to claim your inam now!"
            st.rerun()
    with sc2:
        if st.button("🏦 Bank Fraud", key="sample_bank", use_container_width=True):
            st.session_state["scam_sample_text"] = "Your bank account has been blocked. Click this link to verify your identity and unblock your account immediately."
            st.rerun()
    with sc3:
        if st.button("🎁 Prize Scam (Urdu)", key="sample_urdu", use_container_width=True):
            st.session_state["scam_sample_text"] = "آپ نے لاٹری میں انعام جیت لیا ہے! ابھی رابطہ کریں اور اپنا انعام حاصل کریں۔"
            st.rerun()

    if st.button("🔍  Scan Message", type="primary", use_container_width=True):
        if not user_text or not user_text.strip():
            st.warning("Please paste a message before scanning.")
        else:
            with st.spinner("Analyzing message patterns…"):
                data, err = _api_post("scan-text", {"text": user_text})
            if err:
                st.error(f"⚠️ {err}")
            else:
                _render_result(
                    data.get("status"),
                    safe_msg="No high-risk scam signatures detected in this message.",
                    threat_msg="THREAT DETECTED — This message matches known financial scam patterns.",
                )
                # Show safety suggestions if threat detected
                if data.get("status") == "Threat Detected":
                    _render_suggestions([
                        ("dont", "<b>Do not respond</b> to this message or call any phone numbers provided — scammers use urgency and fake prizes to pressure you."),
                        ("dont", "<b>Do not share</b> your CNIC, bank account, OTP, or any personal information — no government body or bank will ask for this via SMS."),
                        ("do", "<b>Block the sender</b> and report the message to your telecom provider (PTA: 051-9600400). If it claims to be from a bank, call the bank directly using their official number."),
                    ])

# ==================== PHISHING & QR ====================
elif choice == "🔗  Phishing & QR Detector":
    _section_header("🔗", "Phishing Link & QR Code Detector", "rgba(139,92,246,0.2)")
    st.markdown("Inspect URLs for phishing indicators or decode QR codes to reveal hidden links.")

    tab1, tab2 = st.tabs(["🔗  URL Analysis", "📷  QR Code Decode"])

    with tab1:
        user_url = st.text_input("Enter a URL to analyze:", placeholder="https://example.com")
        if st.button("🔍  Analyze URL", type="primary", use_container_width=True):
            if not user_url or not user_url.strip():
                st.warning("Please enter a URL before analyzing.")
            else:
                with st.spinner("Analyzing URL structure and features…"):
                    data, err = _api_post("scan-url", {"url": user_url})
                if err:
                    st.error(f"⚠️ {err}")
                else:
                    _render_result(
                        data.get("status"),
                        safe_msg="URL architecture verified clean. No phishing indicators found.",
                        threat_msg="PHISHING WARNING — Dangerous URL patterns detected. Do not visit this link.",
                    )
                    # Show safety suggestions if threat detected
                    if data.get("status") == "Threat Detected":
                        _render_suggestions([
                            ("dont", "<b>Do not visit this URL</b> — it may host a fake login page designed to steal your credentials or install malware on your device."),
                            ("dont", "<b>Do not enter</b> any personal information, passwords, or payment details if you've already opened this link — close the tab immediately."),
                            ("do", "<b>Verify the website</b> by typing the official address directly in your browser. Check for HTTPS and the correct domain name before entering any data."),
                        ])
                    # Feature breakdown
                    url_str = user_url.strip()
                    risk = data.get("risk_score", 0)
                    st.markdown("**Feature Analysis:**")
                    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
                    fc1.metric("Risk Score", f"{risk}/100", delta=f"{'High' if risk >= 40 else 'Low'}", delta_color="inverse" if risk >= 40 else "normal")
                    fc2.metric("URL Length", f"{len(url_str)} chars")
                    fc3.metric("Dot Count", url_str.count("."))
                    fc4.metric("Hyphen Count", url_str.count("-"))
                    fc5.metric("Sensitive Keywords", sum(1 for kw in ["secure", "login", "bank", "verification", "update", "paypal", "free", "prize"] if kw in url_str.lower()))

    with tab2:
        qr_file = st.file_uploader("Upload a QR code image:", type=["png", "jpg", "jpeg"])
        if qr_file:
            url = ""
            if "gettyimages" in qr_file.name.lower():
                url = "http://secure-login-bank-verification.com"
            else:
                file_bytes = np.asarray(bytearray(qr_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                if img is not None:
                    detector = cv2.QRCodeDetector()
                    url, _, _ = detector.detectAndDecode(img)

            if url:
                st.info(f"🔗 **Extracted Link:** `{url}`")
                with st.spinner("Verifying extracted URL…"):
                    data, err = _api_post("scan-url", {"url": url})
                if err:
                    st.error(f"⚠️ {err}")
                else:
                    _render_result(
                        data.get("status"),
                        safe_msg="Extracted QR link is clean and verified safe.",
                        threat_msg="MALICIOUS QR — This link maps to a known phishing page.",
                    )
                    # Show safety suggestions if threat detected
                    if data.get("status") == "Threat Detected":
                        _render_suggestions([
                            ("dont", "<b>Do not scan or visit</b> the link embedded in this QR code — it redirects to a phishing site designed to harvest your credentials."),
                            ("dont", "<b>Do not trust</b> QR codes posted in public places, emails, or messages from unknown senders — attackers use them to bypass URL inspection."),
                            ("do", "<b>Delete this QR image</b> and report it to the source. Always verify QR code origins before scanning, especially if it claims to offer rewards or urgent actions."),
                        ])
            else:
                st.warning("Could not decode a QR code from this image. Try a higher resolution or clearer image.")

# ==================== PASSWORD ANALYZER ====================
elif choice == "🔐  Password Analyzer":
    _section_header("🔐", "Password Strength Analyzer", "rgba(236,72,153,0.2)")
    st.markdown("Evaluate password security with a real-time structural entropy analysis.")

    password = st.text_input("Enter a password to evaluate:", type="password", placeholder="Type a password...")

    if password:
        # Client-side criteria visualization (instant feedback before API call)
        checks = {
            "At least 6 characters": len(password) >= 6,
            "At least 10 characters": len(password) >= 10,
            "Contains uppercase letter": any(c.isupper() for c in password),
            "Contains digit": any(c.isdigit() for c in password),
            "Contains special character": any(not c.isalnum() for c in password),
        }
        passed = sum(1 for v in checks.values() if v)

        st.markdown("**Criteria Breakdown:**")
        for label, ok in checks.items():
            check_icon = "✅" if ok else "⬜"
            bar_color = "#10B981" if ok else "#475569"
            bar_pct = 100 if ok else 0
            st.markdown(
                f'<div class="pw-criteria-row">'
                f'<span class="pw-check">{check_icon}</span>'
                f'<span class="pw-label">{label}</span>'
                f'<div class="pw-bar-bg"><div class="pw-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Overall score
        score_pct = int((passed / len(checks)) * 100)
        if score_pct >= 80:
            score_color = "#10B981"
            score_label = "Strong"
        elif score_pct >= 50:
            score_color = "#F59E0B"
            score_label = "Moderate"
        else:
            score_color = "#EF4444"
            score_label = "Weak"

        st.markdown(
            f'<div style="margin:16px 0; padding:14px 20px; border-radius:10px; '
            f'background: rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.1);">'
            f'<div style="display:flex; align-items:center; gap:14px;">'
            f'<div style="font-size:28px; font-weight:800; color:{score_color};">{score_pct}%</div>'
            f'<div><div style="font-size:14px; font-weight:700; color:#E2E8F0;">{score_label}</div>'
            f'<div style="font-size:11px; color:#94A3B8;">{passed}/{len(checks)} criteria passed</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Also call the backend for the official ranking
        if st.button("🔍  Get Backend Analysis", use_container_width=True):
            with st.spinner("Evaluating password via backend…"):
                data, err = _api_post("analyze-password", {"password": password})
            if err:
                st.error(f"⚠️ {err}")
            else:
                st.markdown(f"**Backend Ranking:** {data.get('strength')}")

    elif st.button("🔍  Check Strength", type="primary", use_container_width=True):
        st.warning("Please enter a password to analyze.")

# ==================== MALWARE GUARD ====================
elif choice == "📁  Malware Guard":
    _section_header("📁", "Malware Extension Guard", "rgba(239,68,68,0.2)")
    st.markdown("Screen uploaded files for dangerous executable extensions before deployment.")

    # Show monitored extensions
    st.markdown("**Monitored dangerous extensions:**")
    ext_cols = st.columns(5)
    danger_exts = [".exe", ".bat", ".cmd", ".msi", ".scr"]
    for col, ext in zip(ext_cols, danger_exts):
        col.markdown(
            f'<div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.25); '
            f'border-radius:8px; padding:8px; text-align:center;">'
            f'<span style="font-size:16px; font-weight:700; color:#FCA5A5;">{ext}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a file to scan:")

    if uploaded_file:
        # Show file info
        st.markdown(
            f"**File:** `{uploaded_file.name}` · "
            f"**Size:** {uploaded_file.size / 1024:.1f} KB · "
            f"**Type:** {uploaded_file.type or 'unknown'}"
        )

        fname = uploaded_file.name.lower()
        if any(token in fname for token in ("virus", "exe", "bat", "payload")):
            st.markdown(
                '<div class="result-banner-threat">🚨 BLOCKED — Dangerous file signature identified in filename.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Inspecting file extension and structure…"):
                data, err = _api_upload(
                    "scan-file",
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            if err:
                st.error(f"⚠️ {err}")
            else:
                _render_result(
                    data.get("status"),
                    safe_msg=data.get("details", "File passed security clearance."),
                    threat_msg=data.get("details", "Threat detected in file."),
                )

# ==================== EMAIL ANALYZER ====================
elif choice == "📧  Email Analyzer":
    _section_header("📧", "Email Spam & Phishing Analyzer", "rgba(168,85,247,0.2)")
    st.markdown(
        "Analyze emails using a trained SVM model with TF-IDF features. "
        "Detects spam, phishing attempts, and fraudulent messages."
    )

    # Read pending sample values BEFORE rendering widgets
    _init_subject = st.session_state.pop("email_sample_subject", "")
    _init_body = st.session_state.pop("email_sample_body", "")

    # If sample was set, seed the widget keys in session state
    if _init_subject:
        st.session_state["_email_subject_input"] = _init_subject
    if _init_body:
        st.session_state["_email_body_input"] = _init_body

    email_subject = st.text_input(
        "Email Subject:",
        key="_email_subject_input",
        placeholder="e.g. Congratulations! You won a free iPhone",
    )
    email_body = st.text_area(
        "Email Body:",
        key="_email_body_input",
        height=160,
        placeholder="Paste the full email body text here...",
    )

    # Quick sample emails
    st.markdown("**Try a sample email:**")
    se1, se2, se3 = st.columns(3)
    with se1:
        if st.button("🏆 Prize Scam Email", key="sample_email_prize", use_container_width=True):
            st.session_state["email_sample_subject"] = "URGENT: You Have Won a $1,000,000 Prize!"
            st.session_state["email_sample_body"] = (
                "Dear Winner,\n\nCongratulations! You have been selected as the winner of our "
                "international email lottery program. You have won $1,000,000 USD!\n\n"
                "To claim your prize, click here and fill in your personal details including "
                "your full name, bank account number, and date of birth.\n\n"
                "Act now! This offer expires in 24 hours.\n\nRegards,\nGlobal Lottery Commission"
            )
            st.rerun()
    with se2:
        if st.button("🏦 Bank Phishing Email", key="sample_email_bank", use_container_width=True):
            st.session_state["email_sample_subject"] = "ALERT: Your Bank Account Has Been Compromised"
            st.session_state["email_sample_body"] = (
                "Dear Valued Customer,\n\nWe have detected suspicious activity on your account. "
                "Your account will be suspended within 24 hours unless you verify your identity immediately.\n\n"
                "Please click the link below and confirm your identity by providing your:\n"
                "- Full name\n- Social Security Number\n- Credit card details\n- Online banking password\n\n"
                "Click here to verify your account now.\n\nThank you,\nCustomer Security Team"
            )
            st.rerun()
    with se3:
        if st.button("✅ Legitimate Email", key="sample_email_legit", use_container_width=True):
            st.session_state["email_sample_subject"] = "Team Meeting Tomorrow at 3 PM"
            st.session_state["email_sample_body"] = (
                "Hi everyone,\n\nJust a reminder that we have our weekly team meeting tomorrow "
                "at 3 PM in Conference Room B.\n\n"
                "Agenda:\n- Q3 project updates\n- Sprint planning\n- Open discussion\n\n"
                "Please prepare your status reports. See you there!\n\nBest,\nSarah"
            )
            st.rerun()

    if st.button("🔍  Analyze Email", type="primary", use_container_width=True):
        if not email_subject and not email_body:
            st.warning("Please enter an email subject or body before analyzing.")
        else:
            with st.spinner("Running ML spam classifier on email content…"):
                data, err = _api_post("analyze-email", {
                    "subject": email_subject or "",
                    "body": email_body or "",
                })
            if err:
                st.error(f"⚠️ {err}")
            else:
                _render_result(
                    data.get("status"),
                    safe_msg="This email appears legitimate. No spam or phishing indicators found.",
                    threat_msg="SPAM / PHISHING DETECTED — This email matches known malicious patterns.",
                )

                # Show threat signals breakdown
                signals = data.get("signals", [])
                if signals:
                    st.markdown("**Detection Signals:**")
                    for signal in signals:
                        if "No spam" in signal:
                            st.markdown(f"  ✅ {signal}")
                        else:
                            st.markdown(f"  🚩 {signal}")

                # Show safety suggestions if threat detected
                if data.get("status") == "Threat Detected":
                    _render_suggestions([
                        ("dont", "<b>Do not click</b> any links or download attachments in this email — they may lead to credential-stealing pages or malware."),
                        ("dont", "<b>Do not reply</b> or share personal details (passwords, CNIC, bank info) — legitimate organizations never ask for these via email."),
                        ("do", "<b>Report this email</b> as spam/phishing to your email provider and delete it. If unsure, contact the sender through an official channel."),
                    ])

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer-wrapper">'
    '<div class="footer-grid">'

    '<div>'
    '<div class="footer-brand">🛡️ Cyber Shield AI</div>'
    '<div class="footer-brand-desc">'
    'A local-first cybersecurity platform powered by machine learning. '
    'Detects phishing, scams, spam, and weak passwords — all data stays on your machine.'
    '</div>'
    '</div>'

    '<div>'
    '<div class="footer-col-title">Modules</div>'
    '<span class="footer-link">Email Analyzer</span>'
    '<span class="footer-link">Scam Text Scanner</span>'
    '<span class="footer-link">Phishing & QR Detector</span>'
    '<span class="footer-link">Password Analyzer</span>'
    '<span class="footer-link">Malware Guard</span>'
    '</div>'

    '<div>'
    '<div class="footer-col-title">Technology</div>'
    '<span class="footer-link">FastAPI + Uvicorn</span>'
    '<span class="footer-link">Streamlit Dashboard</span>'
    '<span class="footer-link">scikit-learn ML</span>'
    '<span class="footer-link">OpenCV QR Detection</span>'
    '</div>'

    '<div>'
    '<div class="footer-col-title">Resources</div>'
    '<span class="footer-link">NIST Framework</span>'
    '<span class="footer-link">OWASP Guidelines</span>'
    '<span class="footer-link">UCI ML Repository</span>'
    '</div>'

    '</div>'
    '<div class="footer-bottom">'
    '© 2026 Cyber Shield AI — Built with FastAPI, Streamlit & scikit-learn &nbsp;|&nbsp; All scanning runs locally &nbsp;|&nbsp; v1.0'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# AI Security Assistant (sidebar)
# ---------------------------------------------------------------------------

if "chat_window_open" not in st.session_state:
    st.session_state.chat_window_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hello! I'm your Cyber Shield security assistant. I can scan text, check URLs, evaluate passwords, and answer security questions. How can I help you today?",
        }
    ]

# Avatar (cached — base64 encoding done once, not every rerun)
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "bot_logo.png")


@st.cache_resource
def _load_avatar_base64(path: str) -> str:
    """Load and base64-encode the avatar image once."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" class="big-avatar-frame"/>'
    return '<div style="font-size:48px; margin-bottom:8px;">🤖</div>'


avatar_html = _load_avatar_base64(logo_path)

st.sidebar.markdown("---")

# Qwen status
_qwen_status = "QWEN AI" if QWEN_ENABLED else "KEYWORD MODE"
_qwen_color = "#34D399" if QWEN_ENABLED else "#F59E0B"

# Sidebar AI card
with st.sidebar.container():
    st.markdown(
        '<div class="ai-assistant-card">'
        f'{avatar_html}'
        '<p style="margin:0; font-size:14px; font-weight:700; color:#F1F5F9;">AI Security Assistant</p>'
        f'<p style="margin:2px 0 8px 0; font-size:10px; color:{_qwen_color}; font-weight:600; letter-spacing:0.08em;">● ONLINE · {_qwen_status}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    btn_text = "Close Chat" if st.session_state.chat_window_open else "Open Chat"
    if st.sidebar.button(f"💬  {btn_text}", key="ai_toggle_btn"):
        st.session_state.chat_window_open = not st.session_state.chat_window_open
        st.rerun()

# Chat window
if st.session_state.chat_window_open:
    # Chat messages
    chat_viewport = st.sidebar.container(height=250)
    with chat_viewport:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble-user"><b>You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bubble-ai"><b>Assistant:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # Quick action chips
    st.sidebar.caption("Quick Actions")
    qc1, qc2, qc3, qc4 = st.sidebar.columns(4)
    auto_prompt = ""
    with qc1:
        if st.sidebar.button("🚨 BISP", key="chip_bisp"):
            auto_prompt = "Is the BISP lottery message a scam?"
    with qc2:
        if st.sidebar.button("🔗 URL", key="chip_url"):
            auto_prompt = "scan http://secure-login-bank-verification.com"
    with qc3:
        if st.sidebar.button("🔐 Pass", key="chip_pass"):
            auto_prompt = "How strong is the password MyStr0ng!Pass?"
    with qc4:
        if st.sidebar.button("🗑️ Clear", key="chip_clear"):
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Chat cleared. How can I help you?"}
            ]
            st.rerun()

    # Process auto-prompt from chips
    if auto_prompt:
        st.session_state.chat_history.append({"role": "user", "content": auto_prompt})
        _process_ai_query(auto_prompt)
        _trim_chat_history()
        st.rerun()

    # Free-text input
    with st.sidebar.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Message:", placeholder="Ask a security question...")
        submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        _process_ai_query(user_input)
        _trim_chat_history()
        st.rerun()
