from io import BytesIO
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile

from app.api.routes import report, upload
from app.services.task_manager import Job, TaskStatus


def test_report_path_must_be_within_output_directory(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    sibling = tmp_path / "outputs-untrusted" / "report.pdf"
    sibling.parent.mkdir()
    sibling.touch()
    monkeypatch.setattr(report, "ALLOWED_OUTPUT_DIR", str(outputs.resolve()))

    with pytest.raises(HTTPException, match="Access denied"):
        report._validate_report_path(str(sibling))


@pytest.mark.anyio
async def test_report_status_returns_download_url_not_server_path(tmp_path, monkeypatch):
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF")
    task_id = str(uuid4())
    job = Job(id=task_id, status=TaskStatus.COMPLETED, message="done", report_path=str(pdf_path), report_status="ready")

    async def get_job(_task_id):
        return job

    monkeypatch.setattr(report.title_task_manager, "get_job_async", get_job)
    response = await report.get_report_status(task_id)

    assert response == {"status": "ready", "download_url": f"/api/jobs/{task_id}/report"}


@pytest.mark.anyio
async def test_batch_size_validation_limits_total_upload_size(monkeypatch):
    monkeypatch.setattr(upload.settings, "MAX_UPLOAD_SIZE_MB", 1)
    files = [
        UploadFile(filename="one.csv", file=BytesIO(b"a" * 700_000)),
        UploadFile(filename="two.csv", file=BytesIO(b"b" * 700_000)),
    ]

    with pytest.raises(HTTPException) as exc:
        await upload._validate_upload_sizes(files)

    assert exc.value.status_code == 413
