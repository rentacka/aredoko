"""
AreDoko (「あれどこだっけ？」AI記憶庫) FastAPI Web Server
- Runs locally on http://127.0.0.1:1925 (configurable via AREDOKO_PORT)
- CORS enabled for future Chrome/Edge browser extension integration
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine import AreDokoEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

engine = AreDokoEngine(DATA_DIR)

app = FastAPI(title="AreDoko AI Memory", version="1.0.0")

# Allow localhost and browser extensions (future Manifest V3 extension support)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AddItemRequest(BaseModel):
    title: str
    url: str = ""
    source: str = "Memo"
    folder: str = "投げ込みメモ"
    summary: str = ""
    tags: list[str] = []


class SyncBrowserRequest(BaseModel):
    browser: str = "Chrome"


class EnrichRequest(BaseModel):
    max_items: int = 20


class AIConfigRequest(BaseModel):
    preset: str = "aria_local"
    embed_url: str = ""
    embed_model: str = ""
    llm_url: str = ""
    llm_model: str = ""


@app.get("/")
def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(str(index_path))


@app.get("/api/status")
def api_status() -> dict:
    return engine.get_status()


@app.get("/api/config")
def api_get_config() -> dict:
    return engine.get_ai_config_payload()


@app.post("/api/config")
def api_update_config(req: AIConfigRequest) -> dict:
    cfg = engine.update_ai_config(
        preset=req.preset,
        embed_url=req.embed_url,
        embed_model=req.embed_model,
        llm_url=req.llm_url,
        llm_model=req.llm_model,
    )
    return {"ok": True, "config": cfg, "status": engine.get_status()}


@app.get("/api/models")
def api_list_models(url: str = Query(default="")) -> dict:
    target = url.strip() or engine.ai_config["embed_url"]
    return engine.fetch_available_models(target)


@app.post("/api/reindex")
def api_reindex() -> dict:
    return engine.reindex_missing_vectors()


@app.get("/api/search")
def api_search(
    q: str = Query(default=""),
    source: str = Query(default="all"),
    rerank: bool = Query(default=True),
    limit: int = Query(default=40, ge=1, le=200),
) -> dict:
    return engine.search(query=q, source=source, use_rerank=rerank, limit=limit)


@app.post("/api/items")
def api_add_item(req: AddItemRequest) -> dict:
    if not req.title.strip() and not req.url.strip():
        raise HTTPException(status_code=400, detail="タイトルまたはURLを入力してください。")
    item = engine.add_item(
        title=req.title,
        url=req.url,
        source=req.source,
        folder=req.folder,
        summary=req.summary,
        tags=req.tags,
    )
    return {"ok": True, "item": item}


@app.delete("/api/items/{item_id}")
def api_delete_item(item_id: int) -> dict:
    ok = engine.delete_item(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True}


@app.post("/api/sync/browser")
def api_sync_browser(req: SyncBrowserRequest) -> dict:
    if req.browser not in ("Chrome", "Edge", "Brave"):
        raise HTTPException(status_code=400, detail="Unsupported browser")
    return engine.sync_browser(req.browser)


@app.post("/api/sync/html")
async def api_sync_html(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    html_text = raw.decode("utf-8", errors="replace")
    return engine.sync_html(html_text)


@app.post("/api/enrich")
def api_enrich(req: EnrichRequest) -> dict:
    return engine.start_enrichment(max_items=max(1, min(req.max_items, 100)))


@app.get("/api/enrich/status")
def api_enrich_status() -> dict:
    return engine.enrich_state


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AREDOKO_PORT", "1925"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
