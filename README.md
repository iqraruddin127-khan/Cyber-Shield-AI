# CyberShield AI

An AI-powered cybersecurity command center that detects phishing links, scam messages, spam emails, and weak passwords. Runs locally through a FastAPI backend — your data never leaves your machine.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

| Module | Description |
|--------|-------------|
| **Threat Dashboard** | Global security metrics overview with animated radar visualization |
| **Email Analyzer** | Spam & phishing email detection using SVM (LinearSVC) |
| **Scam Text Scanner** | Detect SMS/WhatsApp scams (English, Urdu, Roman Urdu) |
| **Phishing & QR Detector** | URL analysis + QR code decoding with OpenCV |
| **Password Analyzer** | Strength evaluation with criteria breakdown |
| **Malware Guard** | File extension screening for dangerous executables |
| **AI Assistant** | Qwen-powered chatbot for security guidance |

## Tech Stack

- **Backend:** FastAPI + Uvicorn (REST API on port 8000)
- **Frontend:** Streamlit (Interactive dashboard on port 8501)
- **ML Models:** scikit-learn (LinearSVC, XGBoost) + joblib persistence
- **AI Assistant:** Qwen (via DashScope OpenAI-compatible API)
- **Authentication:** Supabase (email/password with session persistence)
- **QR Detection:** OpenCV (`cv2.QRCodeDetector`)

## Project Structure

```
CyberShield AI/
├── Models/                 # Pre-trained ML models (.pkl files)
│   ├── scam_model.pkl
│   ├── scam_vectorizer.pkl
│   ├── phishing_model.pkl
│   └── password_model.pkl
├── extension/              # Chrome extension for URL scanning
│   ├── manifest.json
│   ├── popup.html/js/css
│   └── background.js
├── app.py                  # FastAPI backend
├── dashboard.py            # Streamlit frontend
├── main.py                 # Process launcher
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── README.md
```

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/your-username/cybershield-ai.git
cd cybershield-ai
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Supabase (get from https://supabase.com/dashboard → Settings → API)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key

# DashScope/Qwen AI (get from https://dashscope.console.aliyun.com/)
DASHSCOPE_API_KEY=sk-your-key-here
```

### 3. Set Up Supabase

Run this SQL in your Supabase SQL Editor:

```sql
CREATE TABLE public.profiles (
  id          UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name   TEXT,
  email       TEXT,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
  ON public.profiles FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
  ON public.profiles FOR INSERT WITH CHECK (auth.uid() = id);
```

### 4. Run the Application

```bash
python main.py
```

Or start servers separately:

```bash
# Terminal 1: Backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# Terminal 2: Frontend
python -m streamlit run dashboard.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with model availability status |
| `/api/scan-text` | POST | Scan text for scam patterns |
| `/api/scan-url` | POST | Analyze URL for phishing indicators |
| `/api/analyze-password` | POST | Evaluate password strength |
| `/api/scan-file` | POST | Screen file for dangerous extensions |
| `/api/analyze-email` | POST | Detect spam/phishing in emails |

### Example Request

```bash
curl -X POST http://localhost:8000/api/scan-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://secure-login-bank-verification.tk"}'
```

Response:
```json
{
  "status": "Threat Detected",
  "url": "https://secure-login-bank-verification.tk",
  "risk_score": 75
}
```

## Chrome Extension

The `extension/` folder contains a Manifest V3 Chrome extension for scanning any URL:

1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `extension/` folder
4. Right-click any URL → "CyberShield — Scan this link"

## Security Notes

- **All scanning runs locally** — your data never leaves your machine
- **No secrets in code** — all credentials loaded from environment variables
- **Session persistence** — Supabase auth with automatic token refresh
- **ML model fallback** — keyword heuristics when models are unavailable

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- ML models trained on SMS Spam Collection (UCI ML Repository)
- Phishing detection based on OWASP guidelines
- NIST Cybersecurity Framework for threat detection standards
