---
title: TalentFilter Resume Screener
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🔍 TalentFilter — AI Resume Screener

LLM-powered candidate ranking tool — upload resumes, describe the role, get instant scores with detailed breakdowns.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-FREE-4285F4?style=for-the-badge&logo=google&logoColor=white)

## ✨ Features

- **Multi-LLM Support** — Google Gemini (free), Anthropic Claude, OpenAI GPT
- **Smart Scoring** — AI evaluates Skills Match, Experience, Education & Cultural Fit
- **Adjustable Weights** — Customize how much each dimension contributes
- **Resume Parsing** — Supports PDF, DOCX, and TXT uploads
- **Visual Rankings** — Interactive bar charts, radar charts & detailed breakdowns
- **Export Results** — Download as CSV or JSON
- **Built-in Demo** — Sample resumes included for quick testing

## 🚀 Quick Start

### Local Setup

```bash
git clone https://github.com/duo311/talentfilter-resume-screener.git
cd talentfilter-resume-screener
pip install -r requirements.txt
```

### Configure API Key

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-google-gemini-api-key-here"
```

Get a free key at [aistudio.google.com](https://aistudio.google.com/apikey)

### Run

```bash
streamlit run app.py
```

## 🌐 Live Demo

Try it on [Hugging Face Spaces](https://huggingface.co/spaces/kaushikp26/talentfilter-screener)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| AI/LLM | Google Gemini, Anthropic Claude, OpenAI GPT |
| Charts | Plotly |
| PDF Parsing | PyMuPDF, PyPDF2, pdfplumber |
| DOCX Parsing | python-docx |

## 📁 Project Structure

```
├── app.py                 # Streamlit UI
├── src/
│   ├── extractor.py       # Resume text extraction (PDF/DOCX/TXT)
│   ├── scorer.py          # LLM-based scoring engine
│   ├── screener.py        # Multi-resume orchestrator
│   └── utils.py           # Helpers & utilities
├── requirements.txt
└── README.md
```

## 📄 License

MIT License — free to use, modify, and distribute.

## 👤 Author

**Kaushik P** — [GitHub](https://github.com/duo311)