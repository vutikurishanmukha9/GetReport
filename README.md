# GetReport

## Motto
Turn Your Data Into Professional Reports in Seconds.

## Problem Solving
Data analysis is often time-consuming and requires technical expertise. GetReport solves this by providing an automated platform that ingests raw data files (CSV, Excel), performs robust statistical analysis, generates visualizations, and uses AI to derive actionable business insights. It packages all of this into a professional PDF report, available for instant download, eliminating hours of manual work.

## Tech Stack

### Frontend
- React
- Vite
- TypeScript
- Tailwind CSS
- shadcn/ui
- PapaParse (CSV Parsing)
- SheetJS (Excel Parsing)

### Backend
- Python 3.12+
- FastAPI (Web Framework)
- Pandas & NumPy (Data Processing)
- SciPy & Scikit-learn (Statistical Analysis)
- Matplotlib & Seaborn (Visualization)
- OpenAI API (AI Insights)
- ReportLab (PDF Generation)

## Plan of Action

### Completed
1. Project Restructuring: Separated codebase into distinct Frontend and Backend directories.
2. Frontend Polish: Implemented real client-side file parsing for immediate feedback and removed mock data.
3. Branding: Established "GetReport" identity with custom favicon and metadata.
4. Backend Architecture: Initialized a modular FastAPI application structure.
5. Data Engine: Implemented production-grade data cleaning, type inference, and statistical analysis services.
6. Visualization Engine: Created a system to generate static charts (Heatmaps, Distributions) for reports.
7. AI Integration: Integrated OpenAI to interpret statistical summaries into natural language insights.
9. API Integration: Exposed analysis and reporting services via RESTful endpoints (/upload, /generate-report).
10. End-to-End Testing: Verified complete data flow from upload to PDF generation via automated test scripts.

### Yet to Complete
1. Frontend Integration: Connect the Frontend React components to the Backend API.
2. Deployment: Configure production build steps and deploy to hosting platforms.

## API Reference
- **POST /api/upload**: Accepts a file, cleans it, analyzes it, and returns a JSON dashboard payload.
- **POST /api/generate-report**: Accepts analysis data and charts, returns a downloadable PDF report.
