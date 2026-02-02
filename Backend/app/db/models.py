from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from app.db.base import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)  # UUID string
    filename = Column(String, nullable=True)
    status = Column(String, default="PENDING")         # PENDING, PROCESSING, COMPLETED, FAILED
    progress = Column(Integer, default=0)
    message = Column(String, default="Initialized")
    result = Column(JSON, nullable=True)               # Stores analysis summary/stats
    report_path = Column(String, nullable=True)        # Path to generated PDF
    error = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
