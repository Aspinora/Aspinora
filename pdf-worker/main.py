"""
main.py — PDF-processing worker for the SSC answer-key calculator.

Runs as a normal container (FastAPI + uvicorn), not on Deno/Edge. This
service does exactly one thing: given a URL to a DigiAlm answer-key PDF
(a short-lived Supabase Storage signed URL), download it, parse it with
`parser.parse_pdf`, and return the normalized JSON shape the client's
existing computeScore() already expects. It never talks to Supabase
Storage/DB directly and holds no Supabase credentials — the calling
Supabase Edge Function is responsible for Storage access (creating the
signed URL, deleting the file afterward) and stays the only thing that
needs those secrets.

Auth: a single shared-secret bearer token (WORKER_SHARED_SECRET),
configured as an environment variable here and as a Supabase Edge
Function secret on the other side. This is the only credential this
service needs — it authorizes "accept a parse request from our own Edge
Function," nothing about Supabase itself.
"""

import os
import httpx
import pdfplumber
from io import BytesIO
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from parser import parse_pdf, PdfParseError

app = FastAPI(title="ssc-pdf-worker")

WORKER_SHARED_SECRET = os.environ.get("WORKER_SHARED_SECRET")
MAX_PDF_BYTES = 50 * 1024 * 1024  # 50MB safety cap


class ParseRequest(BaseModel):
    signedUrl: str


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/parse")
async def parse(req: ParseRequest, authorization: str | None = Header(default=None)):
    if not WORKER_SHARED_SECRET:
        raise HTTPException(status_code=500, detail="Worker is misconfigured: WORKER_SHARED_SECRET is not set.")
    expected = f"Bearer {WORKER_SHARED_SECRET}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized.")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(req.signedUrl)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Could not download the PDF (upstream status {resp.status_code}).")
        if len(resp.content) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF is too large.")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Could not download the PDF: {e}")

    try:
        with pdfplumber.open(BytesIO(resp.content)) as pdf:
            result = parse_pdf(pdf)
    except PdfParseError as e:
        # 422: reject the import rather than return a potentially wrong score.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read this PDF ({e}). Make sure it is the DigiAlm answer key / response sheet page saved as PDF.")

    return result
