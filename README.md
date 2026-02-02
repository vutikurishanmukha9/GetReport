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
-   **Components**: `lucide-react`, `recharts` (future), `radix-ui`
-   **State/Query**: TanStack Query (React Query)

### Backend
-   **Framework**: FastAPI (Python 3.12+)
-   **Data Processing**: Pandas, NumPy (Streaming Uploads for large files)
-   **Analysis**: SciPy, Scikit-learn
-   **Persistence**: SQLite + SQLAlchemy (Job History)
-   **AI**: OpenAI API (GPT-4o)
-   **Reporting**: ReportLab Platypus (Professional PDF Generation)

## Key Features
-   **Interactive Data Cleaning**: Two-stage "Human-in-the-Loop" pipeline. Review data quality issues (NaNs, Type Errors) and decide how to fix them (Drop, Fill Mean, etc.) before analysis.
-   **Resilient Architecture**: Streaming uploads handled via tempfiles to prevent RAM exhaustion. Automatic cleanup of old files.
-   **Smart Visualizations**: Trends, Distributions, and Correlations automatically generated.
-   **AI-Powered Insights**: Natural language explanation of trends and anomalies.
-   **Persistent History**: Jobs are saved in a local database. Resume analysis or download past reports anytime.

## Project Structure
-   `/Frontend`: React application source code.
-   `/Backend`: FastAPI application source code.
    -   `/app/services`: Core logic (Data Processing, Reporting, LLM).
    -   `/app/db`: SQLite database models.
    -   `/outputs`: Generated PDF reports (auto-cleaned > 24h).

## Setup & Running

### Backend
```bash
cd Backend
# Activate virtual environment
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
-   `POST /api/upload`: Upload file (Streaming) -> Returns Task ID.
-   `GET /api/status/{task_id}`: Check progress / Get Inspection Report.
-   `POST /api/jobs/{task_id}/analyze`: Submit cleaning rules & Start full analysis.
-   `GET /api/jobs/{task_id}/report`: Download generated PDF.
-   `POST /api/jobs/{task_id}/report`: Regenerate PDF from saved results.
