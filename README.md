# GetReport

## Motto
**Turn Your Data Into Professional Reports in Seconds.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)
[![Celery](https://img.shields.io/badge/Celery-5.3-37814A?logo=celery)](https://docs.celeryq.dev/)
[![WeasyPrint](https://img.shields.io/badge/PDF_Engine-WeasyPrint-FF6600)](https://weasyprint.org/)

---

## Overview
GetReport is an intelligent data analysis platform that transforms raw CSV/Excel files into comprehensive PDF reports with minimal effort. It combines automated statistical analysis, semantic data understanding, and AI-powered insights to deliver actionable business intelligence.

## Problem Solved
Data analysis traditionally requires:
- Technical expertise in statistics and programming
- Hours of manual exploration and cleaning
- Separate tools for visualization and reporting

**GetReport eliminates these barriers** by providing a single platform that:
1. Automatically detects data quality issues
2. Applies intelligent cleaning based on user approval
3. Performs deep statistical analysis
4. Generates publication-ready PDF reports

---

## Key Features

### Dual-Engine PDF Generation (New)
- **Local Dev**: Uses `ReportLab` for fast, lightweight PDF generation without system dependencies.
- **Production**: Uses `WeasyPrint` for HTML/CSS-driven, highly styled reports with professional typography and layouts.
- **Seamless Switch**: Controlled via `PDF_ENGINE` environment variable.

### Trust Foundation (Tier 1)
- **Column Confidence Scores**: Grades every column on Completeness, Consistency, Validity, and Stability.
- **Decision Transparency**: Logs why specific tests (Correlation, Time-Series, Anova) were run or skipped.
- **Semantic Intelligence**: Auto-detects domains (Sales, HR, Finance, Healthcare) to tailor insights.

### Advanced Intelligence (Tier 2)
- **RAG-Powered Insights**: Uses Retrieval-Augmented Generation to provide narrative context without exposing raw confidential rows to AI models.
- **Smart Remediation**: Identifies quality issues (outliers, missing values) and suggests ranked cleaning actions.
- **ML-Ready Recommendations**: Suggests optimal encodings and scalers for future machine learning workflows.

### Robust Statistical Analysis
- **17-Point EDA Checklist**: Rigorous validation including skewness, kurtosis, multicollinearity (VIF)
- **Outlier Detection**: IQR-based flagging with configurable thresholds
- **Correlation Analysis**: Strong pair detection with heatmap visualization
- **Time-Series Detection**: Automatic trend and seasonality analysis when date columns present

---

## Tech Stack

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | React + Vite |
| Language | TypeScript |
| Styling | Tailwind CSS, Shadcn/UI |
| Icons | Lucide React |
| State | TanStack Query |

### Backend
| Component | Technology |
|-----------|------------|
| Framework | FastAPI (Python 3.12+) |
| Task Queue | Celery + Redis |
| Data Processing | Polars, NumPy |
| PDF Engine 1 | ReportLab (Local) |
| PDF Engine 2 | WeasyPrint (Production) |
| Database | SQLite (Local) / PostgreSQL (Prod) |
| AI | OpenAI API (GPT-4o) |

---

## Project Structure
```
GetReport/
├── Frontend/               # React application
│   ├── src/components/     # UI components
│   └── src/pages/          # Route pages
├── Backend/
│   ├── app/
│   │   ├── api/            # FastAPI endpoints
│   │   ├── services/       # Core business logic
│   │   │   ├── analysis.py           # Statistical analysis
│   │   │   ├── report_weasyprint.py  # New HTML/CSS PDF engine
│   │   │   ├── report_generator.py   # PDF factory
│   │   │   ├── semantic_inference.py # Domain/column detection
│   │   │   └── tasks.py              # Celery tasks
│   │   └── db/             # Database models
│   └── Dockerfile          # Production build
└── render.yaml             # Render deployment blueprint
```

---

## Setup and Running

### Backend (Local)
```bash
cd Backend
# Activate virtual environment
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run server (Defaults to PDF_ENGINE=reportlab)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (Local)
```bash
cd Frontend
npm install
npm run dev
```

Access the application at `http://localhost:8080`

---

## Deployment (Production)

The project is optimized for deployment on **Render.com** (or any Docker-based cloud).

### Deployment Artifacts
- **render.yaml**: Blueprint for fully automated deployment (API, Worker, Redis, Frontend).
- **Dockerfile**: Multi-stage build installing WeasyPrint system dependencies (Pango, Cairo) and SSL certs.
- **deployment_guide.md**: Detailed step-by-step instructions.

### Environment Variables
| Variable | Default (Local) | Production |
|----------|-----------------|------------|
| `PDF_ENGINE` | `reportlab` | `weasyprint` |
| `DATABASE_URL` | (empty) -> uses `tasks.db` | `postgres://user:pass@host/db?sslmode=require` |
| `REDIS_URL` | `redis://localhost:6379/0` | `redis://redishost:6379/0` |
| `OPENAI_API_KEY`| (required for AI) | (required for AI) |

---

## Architecture Highlights

### Asynchronous Pipeline
`Client` -> `API` -> `Redis` -> `Celery Worker` -> `PDF Engine`

### Two-Stage Pipeline
1. **Inspection Phase**: Upload triggers immediate data profiling. User reviews quality issues and selects cleaning actions.
2. **Analysis Phase**: Approved rules are applied, full statistical analysis runs, and PDF is generated.

### Performance Optimizations
- **Streaming Uploads**: Large files handled via temp files to prevent RAM exhaustion
- **Polars Engine**: High-performance DataFrame operations (faster than Pandas)
- **Process Pool**: PDF generation offloaded to separate process to avoid blocking API
- **Automatic Cleanup**: Old temp files and reports cleaned after 24 hours

---

## License
MIT License - See LICENSE file for details.
