"""
Chat Route — RAG-powered conversation with analyzed data.
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
import logging

from app.core.limiter import limiter, CHAT_LIMIT
from app.core.config import settings
from app.core.auth import verify_api_key, validate_task_id
from app.services.task_manager import title_task_manager, TaskStatus
from app.services.rag_service import rag_service, SecurityGuard

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/jobs/{task_id}/chat")
@limiter.limit(CHAT_LIMIT)
async def chat_with_job(
    request: Request, task_id: str, body: ChatRequest,
    _auth: None = Depends(verify_api_key),
):
    """
    Chat with the analyzed data (RAG).
    """
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job or job.status != TaskStatus.COMPLETED:
         raise HTTPException(400, "Job is not completed yet.")
    
    # Sanitize and truncate user question (VULN-07: Prompt injection mitigation)
    sanitized_question = SecurityGuard.sanitize_input(body.question)
    if not sanitized_question:
        raise HTTPException(400, "Question cannot be empty.")
    if len(sanitized_question) > settings.MAX_CHAT_QUESTION_LENGTH:
        sanitized_question = sanitized_question[:settings.MAX_CHAT_QUESTION_LENGTH]
    # Pass the full job result (which contains structured analysis & insights)
    # directly into the RAG context to vastly improve model reasoning context.
    response = await rag_service.chat_with_report(task_id, sanitized_question, job_result=job.result)
    return response
