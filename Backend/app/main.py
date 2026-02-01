from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

from app.api import endpoints

app = FastAPI(title=settings.PROJECT_NAME)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"], # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router, prefix="/api", tags=["api"])

@app.get("/health")
def health_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}

@app.get("/")
def root():
    return {"message": "Welcome to GetReport API"}
