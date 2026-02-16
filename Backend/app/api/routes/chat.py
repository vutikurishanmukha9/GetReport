"""
Chat Route — RAG-powered conversation with analyzed data.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging

from app.core.limiter import limiter, CHAT_LIMIT
from app.services.task_manager import title_task_manager, TaskStatus
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/jobs/{task_id}/chat")
@limiter.limit(CHAT_LIMIT)
async def chat_with_job(request: Request, task_id: str, body: ChatRequest):
    """
    Chat with the analyzed data (RAG).
    """
    job = await title_task_manager.get_job_async(task_id)
    if not job or job.status != TaskStatus.COMPLETED:
         raise HTTPException(400, "Job is not completed yet.")
         
    response = await rag_service.chat_with_report(task_id, body.question)
    return response
