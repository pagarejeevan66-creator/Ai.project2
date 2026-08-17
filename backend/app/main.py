from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import jwt
import time
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SECRET = "dev-secret-change-this"
ALGORITHM = "HS256"

app = FastAPI(title="DPI AI Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent.parent / "data" / "docs.json"
ESCALATIONS_PATH = Path(__file__).parent.parent / "data" / "escalations.json"

class LoginRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "en"
    consent_id: Optional[str] = None

class Source(BaseModel):
    id: str
    title: str
    url: Optional[str]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    trace_id: str

class EscalateRequest(BaseModel):
    trace_id: str
    reason: Optional[str] = None

# Load docs and build TF-IDF
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DOCS = json.load(f)

DOC_TEXTS = [d.get("text", "") for d in DOCS]
vectorizer = TfidfVectorizer().fit(DOC_TEXTS)
DOC_VECS = vectorizer.transform(DOC_TEXTS)


def create_token(sub: str) -> str:
    payload = {"sub": sub, "iat": int(time.time()), "exp": int(time.time()) + 3600}
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


def verify_token(auth_header: Optional[str] = Header(None)) -> str:
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    token = auth_header.split(" ", 1)[1]
    try:
        data = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return data.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Token invalid")


@app.post("/v1/auth/login", response_model=AuthResponse)
async def login(body: LoginRequest):
    # Prototype stub: accept any username/password
    token = create_token(body.username)
    return {"access_token": token}


@app.post("/v1/assistant/query", response_model=QueryResponse)
async def assistant_query(body: QueryRequest, current_user: str = Depends(verify_token)):
    q = body.query
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, DOC_VECS)[0]
    top_k = 3
    idxs = np.argsort(sims)[::-1][:top_k]
    sources = []
    combined_answer_parts = []
    for i in idxs:
        doc = DOCS[int(i)]
        sources.append(Source(id=doc.get("id", str(i)), title=doc.get("title", ""), url=doc.get("url")))
        # take short excerpt
        excerpt = doc.get("text", "")[:500]
        combined_answer_parts.append(f"[{doc.get('title','')}] {excerpt}")
    confidence = float(sims[idxs[0]]) if len(sims)>0 else 0.0
    answer = (
        "Note: This is assistive guidance based on retrieved documents. Verify with official sources.\n\n"
        + "\n\n".join(combined_answer_parts)
    )
    trace_id = f"trace-{int(time.time()*1000)}"
    return QueryResponse(answer=answer, sources=sources, confidence=confidence, trace_id=trace_id)


@app.post("/v1/assistant/escalate")
async def escalate(body: EscalateRequest, current_user: str = Depends(verify_token)):
    record = {
        "trace_id": body.trace_id,
        "reason": body.reason,
        "user": current_user,
        "ts": int(time.time()),
    }
    ESCALATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ESCALATIONS_PATH.exists():
        with open(ESCALATIONS_PATH, "r", encoding="utf-8") as f:
            arr = json.load(f)
    else:
        arr = []
    arr.append(record)
    with open(ESCALATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok"}
