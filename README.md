# GetReport

## Motto
**Turn Your Data Into Professional Reports in Seconds.**

## Problem Solving
Data analysis is often time-consuming and requires technical expertise. GetReport solves this by providing an automated platform that ingests raw data files (CSV, Excel), performs robust statistical analysis, generates visualizations, and uses AI to derive actionable business insights. It packages all of this into a professional PDF report, available for instant download.

## Tech Stack

### Frontend
-   **Framework**: React (Vite)
-   **Language**: TypeScript
-   **Styling**: Tailwind CSS, shadcn/ui
-   **Routing**: React Router DOM (Single Page Application)
-   **Animation**: Framer Motion

### Backend
-   **Framework**: FastAPI (Python 3.12+)
-   **Data Processing**: Pandas, NumPy
-   **Analysis**: SciPy, Scikit-learn
-   **AI**: OpenAI API (GPT-4o)
-   **Reporting**: ReportLab (PDF Generation)

## Key Features
-   **Instant Analysis**: Upload any CSV/Excel file and get immediate insights.
-   **Smart Visualizations**: Auto-generated charts based on data types.
-   **AI-Powered Insights**: Natural language explanation of trends and anomalies.
-   **PDF Reports**: Board-ready documents generated on the fly.
-   **Privacy**: Data is processed in-memory and not permanently stored.

## Project Structure
-   `/Frontend`: React application source code.
-   `/Backend`: FastAPI application source code and virtual environment.

## Setup & Running

### Backend
```bash
cd Backend
# active virtual environment (if not active)
.\venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Run Server
py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd Frontend
# Install dependencies
npm install
# Run Dev Server
npm run dev
```

## API Reference
-   `POST /api/upload`: Upload file for analysis.
-   `POST /api/generate-report`: Generate PDF from analysis data.
-   `GET /api/health`: Check server status.
