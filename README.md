# GetReport

## Motto
**Turn Your Data Into Professional Reports in Seconds.**

## Overview
GetReport is an intelligent data analysis platform that transforms raw CSV/Excel files into comprehensive PDF reports with minimal effort. It combines automated statistical analysis, semantic data understanding, and AI-powered insights to deliver actionable business intelligence.

---

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

### Semantic Column Intelligence
- **Domain Detection**: Automatically identifies dataset context (Education, Sales, Healthcare, HR, Finance, Logistics, IoT)
- **Column Role Classification**: Distinguishes between identifiers, metrics, dimensions, and analytical columns
- **Smart Filtering**: Excludes ID/date columns from visualizations while keeping meaningful numeric data

### ML-Ready Feature Engineering
- **Encoding Recommendations**: Suggests One-Hot, Label, or Target encoding for categorical variables
- **Scaling Suggestions**: Recommends StandardScaler, MinMax, RobustScaler, or Log transforms based on data distribution
- **Feature Extraction Ideas**: Proposes new features from dates (year, month, weekday) and text (length, word count)

### Smart Schema Inference
- **Type Correction**: Detects numbers stored as strings, Excel serial dates, and other type mismatches
- **Relationship Detection**: Identifies foreign key relationships and derived columns
- **Quality Checks**: Flags whitespace issues, inconsistent casing, and near-constant columns

### Actionable Recommendations
- **Domain-Specific Guidance**: Tailored suggestions based on detected domain (e.g., "Analyze Attrition Risk" for HR data)
- **Data Quality Priorities**: Ranked recommendations for handling missing values, duplicates, and outliers
- **Analysis Suggestions**: Next-step recommendations for deeper exploration

### Robust Statistical Analysis
- **17-Point EDA Checklist**: Rigorous validation including skewness, kurtosis, multicollinearity (VIF)
- **Outlier Detection**: IQR-based flagging with configurable thresholds
- **Correlation Analysis**: Strong pair detection with heatmap visualization
- **Time-Series Detection**: Automatic trend and seasonality analysis when date columns present

### Professional PDF Reports
- **Executive Summary**: Key metrics and data health overview
- **Data Intelligence Section**: Domain detection, column roles, analysis pairs
- **Multiple Visualizations**: Correlation heatmaps, distribution histograms, bar charts, boxplots
- **Cleaning Documentation**: Full audit trail of data transformations applied

---

## Tech Stack

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | React + Vite |
| Language | TypeScript |
| Styling | Tailwind CSS, shadcn/ui |
| Icons | Lucide React |
| State | TanStack Query |

### Backend
| Component | Technology |
|-----------|------------|
| Framework | FastAPI (Python 3.12+) |
| Data Processing | Polars, NumPy |
| Analysis | SciPy, Scikit-learn |
| PDF Generation | ReportLab Platypus |
| Database | SQLite + SQLAlchemy |
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
│   │   │   ├── visualization.py      # Chart generation
│   │   │   ├── report_generator.py   # PDF creation
│   │   │   ├── semantic_inference.py # Domain/column detection
│   │   │   ├── feature_engineering.py# ML prep suggestions
│   │   │   ├── smart_schema.py       # Type inference
│   │   │   └── recommendations.py    # Actionable insights
│   │   └── db/             # Database models
│   └── outputs/            # Generated PDFs (auto-cleanup >24h)
└── README.md
```

---

## Setup and Running

### Backend
```bash
cd Backend
# Activate virtual environment
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd Frontend
npm install
npm run dev
```

Access the application at `http://localhost:5173`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload file (streaming) - Returns Task ID |
| GET | `/api/status/{task_id}` | Check progress / Get inspection report |
| POST | `/api/jobs/{task_id}/analyze` | Submit cleaning rules and start analysis |
| POST | `/api/jobs/{task_id}/report` | Generate/regenerate PDF from results |
| GET | `/api/jobs/{task_id}/report` | Download generated PDF |
| POST | `/api/jobs/{task_id}/chat` | Chat with analyzed data (RAG) |

---

## Architecture Highlights

### Two-Stage Pipeline
1. **Inspection Phase**: Upload triggers immediate data profiling. User reviews quality issues and selects cleaning actions.
2. **Analysis Phase**: Approved rules are applied, full statistical analysis runs, and PDF is generated.

### Performance Optimizations
- **Streaming Uploads**: Large files handled via temp files to prevent RAM exhaustion
- **Polars Engine**: High-performance DataFrame operations (faster than Pandas)
- **Process Pool**: PDF generation offloaded to separate process to avoid blocking API
- **Automatic Cleanup**: Old temp files and reports cleaned after 24 hours

### Resilient Design
- **Graceful Degradation**: Each analysis section handles errors independently
- **Comprehensive Logging**: Full audit trail for debugging
- **Type Safety**: Python type hints throughout codebase

---

## License
MIT License - See LICENSE file for details.
