import hashlib
import logging
import os

import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cyber_shield")

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Cyber Shield AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = "Models"
SCAM_MODEL_PATH = os.path.join(MODELS_DIR, "scam_model.pkl")
SCAM_VECT_PATH = os.path.join(MODELS_DIR, "scam_vectorizer.pkl")
PHISHING_MODEL_PATH = os.path.join(MODELS_DIR, "phishing_model.pkl")
PASSWORD_MODEL_PATH = os.path.join(MODELS_DIR, "password_model.pkl")
EMAIL_MODEL_PATH = os.path.join(MODELS_DIR, "spam_svm_model_clean (1).pkl")
EMAIL_VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer_clean.pkl")

# Local Urdu + Roman Urdu scam keyword dictionary (shared across endpoints)
SCAM_KEYWORDS = [
    "bisp", "jeeto pakistan", "inam", "lottery",
    "account block", "انعام", "لاٹری", "paisa",
]

PHISHING_URL_KEYWORDS = [
    "login-verification", "secure-bank", "update-account", "verification",
]

# ── Extended phishing heuristics (fallback when ML model unavailable) ──
PHISHING_DOMAIN_KEYWORDS = [
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

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".buzz", ".top",
    ".xyz", ".club", ".work", ".click", ".link", ".icu",
    ".cam", ".rest", ".surf",
]

FREE_HOSTING_DOMAINS = [
    "000webhost", "freehosting", "infinityfree", "awardpace",
    "byethost", "freesite", "wixsite", "weebly", "jimdo",
    "blogspot", "wordpress.com", "tumblr.com", "strikingly",
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl",
    "tiny.cc", "bc.vc", "adf.ly", "shorte.st",
]

def _compute_phishing_score(url: str) -> int:
    """Return a risk score (0-100) based on URL heuristic analysis.
    
    Score >= 40 is considered suspicious / phishing.
    """
    url_lower = url.lower()
    score = 0

    # ── 1. Suspicious domain keywords in URL ──
    kw_hits = sum(1 for kw in PHISHING_DOMAIN_KEYWORDS if kw in url_lower)
    if kw_hits >= 3:
        score += 35
    elif kw_hits >= 2:
        score += 25
    elif kw_hits >= 1:
        score += 10

    # ── 2. Excessive hyphens (phishing domains use many hyphens) ──
    hyphen_count = url_lower.count("-")
    if hyphen_count >= 4:
        score += 20
    elif hyphen_count >= 2:
        score += 10

    # ── 3. Excessive subdomains (e.g. paypal.secure.login.example.com) ──
    # Extract host part
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        host = url
    subdomain_dots = host.count(".") - 1  # subtract the TLD dot
    if subdomain_dots >= 3:
        score += 15
    elif subdomain_dots >= 2:
        score += 8

    # ── 4. Suspicious TLDs ──
    for tld in SUSPICIOUS_TLDS:
        if url_lower.endswith(tld) or url_lower.endswith(tld + "/"):
            score += 20
            break

    # ── 5. Free hosting / website builders ──
    for fh in FREE_HOSTING_DOMAINS:
        if fh in url_lower:
            score += 15
            break

    # ── 6. URL shorteners (hides true destination) ──
    for us in URL_SHORTENERS:
        if us in url_lower:
            score += 20
            break

    # ── 7. IP address instead of domain name ──
    import re
    if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_lower):
        score += 25

    # ── 8. Missing HTTPS ──
    if not url_lower.startswith("https"):
        score += 10

    # ── 9. Very long URL (phishing URLs tend to be long) ──
    if len(url) > 75:
        score += 10
    if len(url) > 120:
        score += 5

    # ── 10. @ symbol in URL (redirects to different host) ──
    if "@" in url_lower:
        score += 25

    # ── 11. Excessive dots ──
    if url_lower.count(".") >= 5:
        score += 10

    # ── 12. Numbers in domain (suspicious for brand impersonation) ──
    domain_part = host.split(".")[0] if host else ""
    if domain_part and any(c.isdigit() for c in domain_part) and any(c.isalpha() for c in domain_part):
        score += 5

    return min(score, 100)

# ---------------------------------------------------------------------------
# Model loading with availability tracking
# ---------------------------------------------------------------------------
scam_model = None
scam_vectorizer = None
phishing_model = None
email_model = None
email_vectorizer = None

scam_model_available = False
phishing_model_available = False
email_model_available = False

try:
    scam_model = joblib.load(SCAM_MODEL_PATH)
    scam_vectorizer = joblib.load(SCAM_VECT_PATH)
    scam_model_available = True
    logger.info("Scam ML models loaded successfully.")
except Exception:
    logger.exception("Failed to load scam models — falling back to keyword heuristics.")

try:
    phishing_model = joblib.load(PHISHING_MODEL_PATH)
    phishing_model_available = True
    logger.info("Phishing ML model loaded successfully.")
except Exception:
    logger.exception("Failed to load phishing model — falling back to keyword heuristics.")

try:
    email_model = joblib.load(EMAIL_MODEL_PATH)
    email_vectorizer = joblib.load(EMAIL_VECT_PATH)
    email_model_available = True
    logger.info("Email spam ML model loaded successfully.")
except Exception:
    logger.exception("Failed to load email spam model — falling back to keyword heuristics.")

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TextPayload(BaseModel):
    text: str


class UrlPayload(BaseModel):
    url: str


class PasswordPayload(BaseModel):
    password: str


class EmailPayload(BaseModel):
    subject: str
    body: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _has_scam_keywords(text: str) -> bool:
    """Return True if any known scam keyword appears in the text."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SCAM_KEYWORDS)


def _has_phishing_keywords(url: str) -> bool:
    """Return True if the URL triggers phishing heuristics (score >= 40)."""
    return _compute_phishing_score(url) >= 40


# ---------------------------------------------------------------------------
# Response caching (LRU cache for repeated predictions)
# ---------------------------------------------------------------------------
_text_scan_cache: dict[str, dict] = {}
_url_scan_cache: dict[str, dict] = {}
_password_cache: dict[str, dict] = {}
_CACHE_MAX = 512


def _cache_get(cache: dict, key: str) -> dict | None:
    """Get from cache, returning None if missing."""
    return cache.get(key)


def _cache_set(cache: dict, key: str, value: dict) -> None:
    """Set in cache, evicting oldest if full."""
    if len(cache) >= _CACHE_MAX:
        oldest = next(iter(cache))
        del cache[oldest]
    cache[key] = value


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Health check used by the launcher and dashboard to verify backend readiness."""
    return {
        "status": "ok",
        "scam_model_available": scam_model_available,
        "phishing_model_available": phishing_model_available,
        "email_model_available": email_model_available,
    }


@app.post("/api/scan-text")
def scan_text(payload: TextPayload):
    # Check cache first
    text_hash = hashlib.md5(payload.text.encode()).hexdigest()
    cached = _cache_get(_text_scan_cache, text_hash)
    if cached:
        return cached

    try:
        if scam_model_available:
            vectorized_text = scam_vectorizer.transform([payload.text])
            prediction = scam_model.predict(vectorized_text)
            model_threat = prediction[0] == "spam"
        else:
            model_threat = False

        is_local_scam = _has_scam_keywords(payload.text)
        status = "Threat Detected" if (model_threat or is_local_scam) else "Safe"
        result = {"status": status}
        _cache_set(_text_scan_cache, text_hash, result)
        return result
    except Exception:
        logger.exception("Error in scan_text — falling back to keyword check.")
        return {"status": "Threat Detected" if _has_scam_keywords(payload.text) else "Safe"}


@app.post("/api/scan-url")
def scan_url(payload: UrlPayload):
    # Check cache first
    cached = _cache_get(_url_scan_cache, payload.url)
    if cached:
        return cached

    try:
        url_str = str(payload.url)
        phishing_score = _compute_phishing_score(url_str)

        if phishing_model_available:
            features = [[
                len(url_str),
                url_str.count("."),
                url_str.count("-"),
                1 if "secure" in url_str.lower() else 0,
                1 if "login" in url_str.lower() else 0,
                1 if "bank" in url_str.lower() else 0,
            ]]
            prediction = phishing_model.predict(features)
            model_threat = int(prediction[0]) == 1
        else:
            model_threat = False

        # Combine model + heuristic scoring
        heuristic_threat = phishing_score >= 40
        status = "Threat Detected" if (model_threat or heuristic_threat) else "Safe"
        result = {"status": status, "url": payload.url, "risk_score": phishing_score}
        _cache_set(_url_scan_cache, payload.url, result)
        return result
    except Exception:
        logger.exception("Error in scan_url — falling back to heuristic check.")
        score = _compute_phishing_score(payload.url)
        status = "Threat Detected" if score >= 40 else "Safe"
        return {"status": status, "url": payload.url, "risk_score": score}


@app.post("/api/analyze-password")
def analyze_password(payload: PasswordPayload):
    # Check cache first
    cached = _cache_get(_password_cache, payload.password)
    if cached:
        return cached

    password = payload.password
    if len(password) < 6:
        result = {"strength": "Weak 🔴"}
    else:
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_spec = any(not c.isalnum() for c in password)

        if has_upper and has_digit and has_spec and len(password) >= 10:
            result = {"strength": "Strong 🟢"}
        else:
            result = {"strength": "Medium 🟡"}

    _cache_set(_password_cache, payload.password, result)
    return result


# Email spam keyword heuristics (fallback when model unavailable)
EMAIL_SPAM_KEYWORDS = [
    "win", "winner", "won", "free", "prize", "claim", "urgent", "act now",
    "limited time", "click here", "congratulations", "you have been selected",
    "million", "cash", "bonus", "offer expires", "no obligation",
    "risk free", "guaranteed", "earn money", "work from home",
]


def _has_email_spam_keywords(text: str) -> bool:
    """Return True if the text contains multiple spam indicators."""
    text_lower = text.lower()
    matches = sum(1 for kw in EMAIL_SPAM_KEYWORDS if kw in text_lower)
    return matches >= 2  # require at least 2 spam signals


@app.post("/api/analyze-email")
def analyze_email(payload: EmailPayload):
    """Analyze an email (subject + body) for spam/phishing indicators."""
    combined_text = f"{payload.subject} {payload.body}"
    try:
        if email_model_available:
            vectorized = email_vectorizer.transform([combined_text])
            prediction = email_model.predict(vectorized)
            model_threat = int(prediction[0]) == 1
        else:
            model_threat = False

        keyword_threat = _has_email_spam_keywords(combined_text)

        if model_threat or keyword_threat:
            status = "Threat Detected"
        else:
            status = "Safe"

        # Risk score: combine model + keyword signals
        signals = []
        if model_threat:
            signals.append("ML model classified as spam")
        if keyword_threat:
            signals.append("Multiple spam keywords detected")
        if any(kw in combined_text.lower() for kw in ["click here", "click now", "act now"]):
            signals.append("Contains urgent call-to-action")
        if any(kw in combined_text.lower() for kw in ["verify", "confirm identity", "update account"]):
            signals.append("Potential phishing — identity request")

        return {
            "status": status,
            "signals": signals if signals else ["No spam indicators found"],
        }
    except Exception:
        logger.exception("Error in analyze_email — falling back to keyword check.")
        keyword_threat = _has_email_spam_keywords(combined_text)
        return {
            "status": "Threat Detected" if keyword_threat else "Safe",
            "signals": ["Keyword fallback (model error)"] if keyword_threat else ["No spam indicators found"],
        }


@app.post("/api/scan-file")
async def scan_file(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    _, ext = os.path.splitext(filename.lower())
    dangerous_extensions = [".exe", ".bat", ".cmd", ".msi", ".scr"]
    if ext in dangerous_extensions:
        return {"status": "Threat Detected", "details": f"Dangerous executable script ({ext}) flagged."}
    return {"status": "Safe", "details": "File layout clearance passed."}
