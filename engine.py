"""
AreDoko (「あれどこだっけ？」AI記憶庫) Core Engine
- Cross-platform Chromium (Chrome/Edge/Brave) Bookmarks JSON reader (read-only)
- Netscape Bookmark HTML importer (Safari/Firefox/Chrome universal export)
- SQLite metadata store + NumPy (.npz) L2-normalized dense vector store (per-model namespace)
- Multi-provider OpenAI-compatible local AI support:
  Aria Local (Port 1919/1918) / Ollama (11434) / LM Studio (1234) / Custom llama-server
- Hybrid search: Dense (Embeddings) + Sparse (JP 2-gram + word BM25)
  + optional local LLM rerank
- Background page fetch + local LLM AI summary & tag enrichment
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx
import numpy as np
from bs4 import BeautifulSoup

EMBED_BASE_URL = os.environ.get("AREDOKO_EMBED_URL", "http://127.0.0.1:1919")
EMBED_MODEL_DEFAULT = os.environ.get("AREDOKO_EMBED_MODEL", "")
LLM_BASE_URL = os.environ.get("AREDOKO_LLM_URL", "http://127.0.0.1:1918")
LLM_MODEL_DEFAULT = os.environ.get("AREDOKO_LLM_MODEL", "")
EMBED_DIM = 768

AI_PRESETS: dict[str, dict[str, str]] = {
    "aria_local": {
        "label": "Aria Local (1919+1918)",
        "embed_url": EMBED_BASE_URL,
        "embed_model": EMBED_MODEL_DEFAULT,
        "llm_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL_DEFAULT,
    },
    "ollama": {
        "label": "Ollama (11434)",
        "embed_url": "http://127.0.0.1:11434",
        "embed_model": "nomic-embed-text",
        "llm_url": "http://127.0.0.1:11434",
        "llm_model": "gemma3:4b",
    },
    "lmstudio": {
        "label": "LM Studio (1234)",
        "embed_url": "http://127.0.0.1:1234",
        "embed_model": "",
        "llm_url": "http://127.0.0.1:1234",
        "llm_model": "",
    },
    "custom": {
        "label": "カスタム (OpenAI互換)",
        "embed_url": "http://127.0.0.1:1919",
        "embed_model": "",
        "llm_url": "http://127.0.0.1:1918",
        "llm_model": "",
    },
}


def normalize_base_url(url: str) -> str:
    """Normalize user-entered endpoint URL to a clean base URL without /v1/... suffix."""
    u = (url or "").strip().rstrip("/")
    for suffix in (
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/models",
        "/api/embeddings",
        "/api/embed",
        "/api/tags",
        "/v1",
    ):
        if u.lower().endswith(suffix):
            u = u[: -len(suffix)].rstrip("/")
            break
    return u


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by thinking LLMs."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    return cleaned.strip()


def _now_ts() -> float:
    return time.time()


def _extract_domain(url: str) -> str:
    if not url:
        return "memo / local"
    try:
        parsed = urlparse(url)
        return parsed.netloc or "memo / local"
    except Exception:
        return "memo / local"


def _tokenize_jp_en(text: str) -> list[str]:
    """Lightweight Japanese char 2-gram + ASCII/word tokenizer for zero-dependency BM25."""
    if not text:
        return []
    low = text.lower()
    # ASCII / alphanumeric words
    words = re.findall(r"[a-z0-9_+#.-]{2,}", low)
    # Japanese / CJK / Hangul segments -> 2-grams (plus 1-gram if single char)
    jp_segs = re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7afー]+", low)
    grams: list[str] = []
    for seg in jp_segs:
        if len(seg) == 1:
            grams.append(seg)
        else:
            for i in range(len(seg) - 1):
                grams.append(seg[i : i + 2])
    return words + grams


def _make_item_key(source: str, url: str, title: str) -> str:
    base = url.strip() if url.strip() else f"memo::{title.strip()}"
    return hashlib.sha256(f"{source}::{base}".encode("utf-8")).hexdigest()[:24]


def detect_browser_bookmark_paths() -> dict[str, list[Path]]:
    """Detect Chromium-based browser Bookmarks JSON files across Windows, macOS, and Linux."""
    sys_name = platform.system()
    home = Path.home()
    candidates: dict[str, list[Path]] = {"Chrome": [], "Edge": [], "Brave": []}

    if sys_name == "Windows":
        local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        roots = {
            "Chrome": local / "Google" / "Chrome" / "User Data",
            "Edge": local / "Microsoft" / "Edge" / "User Data",
            "Brave": local / "BraveSoftware" / "Brave-Browser" / "User Data",
        }
    elif sys_name == "Darwin":
        app_sup = home / "Library" / "Application Support"
        roots = {
            "Chrome": app_sup / "Google" / "Chrome",
            "Edge": app_sup / "Microsoft Edge",
            "Brave": app_sup / "BraveSoftware" / "Brave-Browser",
        }
    else:
        cfg = home / ".config"
        roots = {
            "Chrome": cfg / "google-chrome",
            "Edge": cfg / "microsoft-edge",
            "Brave": cfg / "BraveSoftware" / "Brave-Browser",
        }

    for browser, root in roots.items():
        if not root.exists():
            continue
        # Check Default and Profile 1..5
        profiles = ["Default"] + [f"Profile {i}" for i in range(1, 6)]
        for prof in profiles:
            bm = root / prof / "Bookmarks"
            if bm.is_file():
                candidates[browser].append(bm)
    return candidates


def parse_chromium_bookmarks_file(path: Path, source: str) -> list[dict[str, str]]:
    """Read Chromium Bookmarks JSON in read-only mode without touching original file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    data = json.loads(raw)
    roots = data.get("roots", {})
    results: list[dict[str, str]] = []

    def walk(node: dict[str, Any], folder_trail: list[str]) -> None:
        ntype = node.get("type")
        name = (node.get("name") or "").strip()
        if ntype == "url":
            url = (node.get("url") or "").strip()
            if url.startswith(("http://", "https://")):
                folder_str = " / ".join(p for p in folder_trail if p) or "未整理"
                results.append(
                    {
                        "source": source,
                        "title": name or url,
                        "url": url,
                        "folder": folder_str,
                    }
                )
        elif ntype == "folder":
            next_trail = folder_trail + ([name] if name else [])
            for child in node.get("children", []):
                walk(child, next_trail)

    for root_key, root_node in roots.items():
        if isinstance(root_node, dict):
            walk(root_node, [])
    return results


def parse_netscape_html_bookmarks(html_text: str, source: str = "HTML") -> list[dict[str, str]]:
    """Parse standard Netscape Bookmark HTML exported from Chrome, Edge, Safari, or Firefox."""
    soup = BeautifulSoup(html_text, "html.parser")
    results: list[dict[str, str]] = []
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href.startswith(("http://", "https://")):
            continue
        title = a.get_text(strip=True) or href
        # Find nearest preceding H3 folder header if available
        folder = "HTML取込"
        dl = a.find_parent("dl")
        if dl:
            prev_h3 = dl.find_previous_sibling("h3")
            if prev_h3:
                folder = prev_h3.get_text(strip=True) or folder
        results.append(
            {
                "source": source,
                "title": title,
                "url": href,
                "folder": folder,
            }
        )
    return results


class AreDokoEngine:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "aredoko.db"
        self.config_path = self.data_dir / "ai_config.json"
        self._lock = threading.RLock()

        # AI provider configuration (persisted to data/ai_config.json)
        self.ai_config: dict[str, str] = self._load_ai_config()

        # In-memory vector cache: item_id -> normalized float32 vector (dynamic dimension)
        self._vec_ids: np.ndarray = np.array([], dtype=np.int64)
        self._vec_mat: np.ndarray = np.zeros((0, 0), dtype=np.float32)

        # Background enrich state
        self.enrich_state: dict[str, Any] = {
            "running": False,
            "done": 0,
            "total": 0,
            "last_title": "",
            "error": "",
        }

        self._init_db()
        self._load_vectors()

    @property
    def vec_path(self) -> Path:
        """Separate vector file per (embed_url, embed_model) namespace while preserving legacy embeddings.npz."""
        default_url = normalize_base_url(EMBED_BASE_URL)
        cur_url = normalize_base_url(self.ai_config.get("embed_url", ""))
        cur_model = (self.ai_config.get("embed_model") or "").strip()
        if cur_url == default_url and cur_model == EMBED_MODEL_DEFAULT:
            return self.data_dir / "embeddings.npz"
        ns_key = hashlib.sha256(f"{cur_url}::{cur_model}".encode("utf-8")).hexdigest()[:12]
        return self.data_dir / f"embeddings_{ns_key}.npz"

    def _default_ai_config(self) -> dict[str, str]:
        p = AI_PRESETS["aria_local"]
        return {
            "preset": "aria_local",
            "embed_url": normalize_base_url(p["embed_url"]),
            "embed_model": p["embed_model"],
            "llm_url": normalize_base_url(p["llm_url"]),
            "llm_model": p["llm_model"],
        }

    def _load_ai_config(self) -> dict[str, str]:
        cfg = self._default_ai_config()
        if self.config_path.is_file():
            try:
                loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    preset = str(loaded.get("preset") or cfg["preset"])
                    if preset not in AI_PRESETS:
                        preset = "custom"
                    cfg["preset"] = preset
                    if "embed_url" in loaded and str(loaded["embed_url"]).strip():
                        cfg["embed_url"] = normalize_base_url(str(loaded["embed_url"]))
                    if "embed_model" in loaded:
                        cfg["embed_model"] = str(loaded.get("embed_model") or "").strip()
                    if "llm_url" in loaded and str(loaded["llm_url"]).strip():
                        cfg["llm_url"] = normalize_base_url(str(loaded["llm_url"]))
                    if "llm_model" in loaded:
                        cfg["llm_model"] = str(loaded.get("llm_model") or "").strip()
            except Exception:
                pass
        return cfg

    def _save_ai_config(self) -> None:
        tmp = self.data_dir / "ai_config.tmp.json"
        tmp.write_text(json.dumps(self.ai_config, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.config_path)

    def get_ai_config_payload(self) -> dict[str, Any]:
        return {
            "config": dict(self.ai_config),
            "presets": AI_PRESETS,
            "vector_file": self.vec_path.name,
            "vector_dim": int(self._vec_mat.shape[1]) if self._vec_mat.ndim == 2 else 0,
        }

    def update_ai_config(
        self,
        preset: str = "aria_local",
        embed_url: str = "",
        embed_model: str = "",
        llm_url: str = "",
        llm_model: str = "",
    ) -> dict[str, str]:
        if preset not in AI_PRESETS:
            preset = "custom"
        base = AI_PRESETS[preset]
        new_cfg = {
            "preset": preset,
            "embed_url": normalize_base_url(embed_url) or normalize_base_url(base["embed_url"]),
            "embed_model": embed_model.strip(),
            "llm_url": normalize_base_url(llm_url) or normalize_base_url(base["llm_url"]),
            "llm_model": llm_model.strip(),
        }
        with self._lock:
            old_vec_path = self.vec_path
            self.ai_config = new_cfg
            self._save_ai_config()
            if self.vec_path != old_vec_path:
                self._load_vectors()
        return dict(self.ai_config)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_key TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL DEFAULT '',
                    folder TEXT NOT NULL DEFAULT '未整理',
                    summary TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '[]',
                    domain TEXT NOT NULL DEFAULT '',
                    enriched INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_items_source ON items(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_items_enriched ON items(enriched)")
            conn.commit()

    def _load_vectors(self) -> None:
        with self._lock:
            vp = self.vec_path
            if vp.is_file():
                try:
                    data = np.load(str(vp))
                    ids = data["ids"].astype(np.int64)
                    mat = data["mat"].astype(np.float32)
                    if mat.ndim == 2 and len(ids) == mat.shape[0]:
                        self._vec_ids = ids
                        self._vec_mat = mat
                        return
                except Exception:
                    pass
            self._vec_ids = np.array([], dtype=np.int64)
            self._vec_mat = np.zeros((0, 0), dtype=np.float32)

    def _save_vectors(self) -> None:
        with self._lock:
            vp = self.vec_path
            tmp = self.data_dir / f"{vp.stem}.tmp.npz"
            np.savez(str(tmp), ids=self._vec_ids, mat=self._vec_mat)
            tmp.replace(vp)

    # ---------- Local AI Server Helpers ----------
    def _probe_server(self, base_url: str) -> bool:
        if not base_url:
            return False
        base = normalize_base_url(base_url)
        with httpx.Client(timeout=1.5) as client:
            for path in ("/v1/models", "/health", "/api/tags"):
                try:
                    r = client.get(f"{base}{path}")
                    if r.status_code == 200:
                        return True
                except Exception:
                    continue
        return False

    def fetch_available_models(self, base_url: str) -> dict[str, Any]:
        """Query OpenAI /v1/models or Ollama /api/tags to list available model names."""
        base = normalize_base_url(base_url)
        if not base:
            return {"ok": False, "models": [], "error": "URLが空です。"}
        models: list[str] = []
        seen: set[str] = set()
        with httpx.Client(timeout=3.0) as client:
            # 1. OpenAI-compatible /v1/models (Ollama / LM Studio / llama-server / vLLM)
            try:
                r = client.get(f"{base}/v1/models")
                if r.status_code == 200:
                    data = r.json()
                    for item in data.get("data", []):
                        mid = str(item.get("id") or "").strip()
                        if mid and mid not in seen:
                            seen.add(mid)
                            models.append(mid)
            except Exception:
                pass
            # 2. Ollama native /api/tags fallback
            if not models:
                try:
                    r = client.get(f"{base}/api/tags")
                    if r.status_code == 200:
                        data = r.json()
                        for item in data.get("models", []):
                            mid = str(item.get("name") or "").strip()
                            if mid and mid not in seen:
                                seen.add(mid)
                                models.append(mid)
                except Exception:
                    pass
        if models:
            return {"ok": True, "url": base, "models": models}
        alive = self._probe_server(base)
        if alive:
            return {"ok": True, "url": base, "models": [], "note": "サーバーは稼働中ですがモデル一覧が空です（空欄のまま利用可能）。"}
        return {"ok": False, "url": base, "models": [], "error": f"{base} に接続できませんでした。"}

    def check_servers(self) -> dict[str, bool]:
        embed_url = self.ai_config.get("embed_url", "")
        llm_url = self.ai_config.get("llm_url", "")
        ok_embed = self._probe_server(embed_url)
        ok_llm = ok_embed if normalize_base_url(llm_url) == normalize_base_url(embed_url) else self._probe_server(llm_url)
        return {
            "embed": ok_embed,
            "llm": ok_llm,
            # Legacy keys kept for backward compatibility
            "embed_1919": ok_embed,
            "llm_1918": ok_llm,
        }

    def _embed_texts(self, texts: list[str]) -> np.ndarray | None:
        if not texts:
            return None
        embed_url = normalize_base_url(self.ai_config.get("embed_url", ""))
        embed_model = (self.ai_config.get("embed_model") or "").strip()
        if not embed_url:
            return None
        try:
            all_vecs: list[list[float]] = []
            batch_size = 32
            with httpx.Client(timeout=25.0) as client:
                for i in range(0, len(texts), batch_size):
                    batch = [t[:1200] for t in texts[i : i + batch_size]]
                    req_body: dict[str, Any] = {"input": batch}
                    if embed_model:
                        req_body["model"] = embed_model
                    resp = client.post(f"{embed_url}/v1/embeddings", json=req_body)
                    resp.raise_for_status()
                    payload = resp.json()
                    for row in payload["data"]:
                        all_vecs.append(row["embedding"])
            mat = np.asarray(all_vecs, dtype=np.float32)
            if mat.ndim != 2 or mat.shape[0] != len(texts):
                return None
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return mat / norms
        except Exception:
            return None

    def _item_embed_text(self, row: dict[str, Any]) -> str:
        tags = row.get("tags") or []
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        tag_str = " ".join(f"#{t}" for t in tags)
        parts = [
            row.get("title", ""),
            row.get("summary", ""),
            tag_str,
            row.get("folder", ""),
            row.get("domain", ""),
        ]
        return " | ".join(p for p in parts if p)

    def _upsert_vectors_for_ids(self, item_ids: list[int]) -> int:
        if not item_ids:
            return 0
        placeholders = ",".join("?" for _ in item_ids)
        with self._connect() as conn:
            rows = [
                dict(r)
                for r in conn.execute(
                    f"SELECT * FROM items WHERE id IN ({placeholders})", item_ids
                ).fetchall()
            ]
        if not rows:
            return 0
        texts = [self._item_embed_text(r) for r in rows]
        vecs = self._embed_texts(texts)
        if vecs is None:
            return 0
        new_dim = int(vecs.shape[1])

        with self._lock:
            existing: dict[int, np.ndarray] = {}
            if self._vec_mat.ndim == 2 and self._vec_mat.shape[1] == new_dim:
                for idx, vid in enumerate(self._vec_ids.tolist()):
                    existing[int(vid)] = self._vec_mat[idx]
            for r, v in zip(rows, vecs):
                existing[int(r["id"])] = v
            sorted_ids = sorted(existing.keys())
            self._vec_ids = np.asarray(sorted_ids, dtype=np.int64)
            self._vec_mat = np.stack([existing[k] for k in sorted_ids], axis=0).astype(np.float32)
            self._save_vectors()
        return len(rows)

    def reindex_missing_vectors(self) -> dict[str, Any]:
        """Embed all items that are missing vectors in the active model namespace."""
        with self._lock:
            have_ids = set(self._vec_ids.tolist())
        with self._connect() as conn:
            all_ids = [int(r["id"]) for r in conn.execute("SELECT id FROM items").fetchall()]
        missing_ids = [i for i in all_ids if i not in have_ids]
        if not missing_ids:
            return {
                "ok": True,
                "embedded": 0,
                "total": len(all_ids),
                "vectors": int(len(self._vec_ids)),
                "message": "すべての記憶はすでに現在のモデルでベクトル化済みです✨",
            }
        embedded = self._upsert_vectors_for_ids(missing_ids)
        if embedded == 0:
            return {
                "ok": False,
                "embedded": 0,
                "total": len(all_ids),
                "vectors": int(len(self._vec_ids)),
                "message": f"Embedding サーバー ({self.ai_config['embed_url']}) に接続できなかったか、モデル名が正しくありません。",
            }
        return {
            "ok": True,
            "embedded": embedded,
            "total": len(all_ids),
            "vectors": int(len(self._vec_ids)),
            "message": f"{embedded}件の記憶を現在のモデルでベクトル化しました！",
        }

    # ---------- CRUD & Sync ----------
    def add_item(
        self,
        title: str,
        url: str = "",
        source: str = "Memo",
        folder: str = "投げ込みメモ",
        summary: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        clean_title = title.strip()
        clean_url = url.strip()
        if not clean_url and re.match(r"^https?://\S+$", clean_title, re.I):
            clean_url = clean_title
        if not clean_title:
            clean_title = clean_url or "無題のメモ"

        if source == "Memo" and ("x.com/" in clean_url or "twitter.com/" in clean_url):
            source = "X"
            folder = "X 保存"

        item_key = _make_item_key(source, clean_url, clean_title)
        domain = _extract_domain(clean_url)
        now = _now_ts()
        tags_json = json.dumps(tags or ([] if clean_url else ["雑メモ"]), ensure_ascii=False)

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO items (item_key, source, title, url, folder, summary, tags, domain, enriched, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_key) DO UPDATE SET
                    title = excluded.title,
                    folder = excluded.folder,
                    updated_at = excluded.updated_at
                """,
                (
                    item_key,
                    source,
                    clean_title,
                    clean_url,
                    folder,
                    summary,
                    tags_json,
                    domain,
                    1 if summary else 0,
                    now,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM items WHERE item_key = ?", (item_key,)).fetchone()
        item_dict = self._format_row(dict(row))
        self._upsert_vectors_for_ids([item_dict["id"]])
        return item_dict

    def delete_item(self, item_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
            conn.commit()
            deleted = cur.rowcount > 0
        if deleted:
            with self._lock:
                mask = self._vec_ids != item_id
                self._vec_ids = self._vec_ids[mask]
                self._vec_mat = self._vec_mat[mask]
                self._save_vectors()
        return deleted

    def bulk_upsert_items(self, records: list[dict[str, str]]) -> dict[str, int]:
        if not records:
            return {"inserted": 0, "updated": 0, "embedded": 0}
        now = _now_ts()
        inserted = 0
        updated = 0
        new_ids: list[int] = []

        with self._connect() as conn:
            existing_keys = {
                r["item_key"]: r["id"]
                for r in conn.execute("SELECT id, item_key FROM items").fetchall()
            }
            for rec in records:
                source = rec.get("source", "Chrome")
                url = rec.get("url", "").strip()
                title = rec.get("title", "").strip() or url
                folder = rec.get("folder", "未整理")
                if "x.com/" in url or "twitter.com/" in url:
                    source = "X"
                key = _make_item_key(source, url, title)
                domain = _extract_domain(url)
                if key in existing_keys:
                    conn.execute(
                        "UPDATE items SET title = ?, folder = ?, updated_at = ? WHERE id = ?",
                        (title, folder, now, existing_keys[key]),
                    )
                    updated += 1
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO items (item_key, source, title, url, folder, summary, tags, domain, enriched, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, '', '[]', ?, 0, ?, ?)
                        """,
                        (key, source, title, url, folder, domain, now, now),
                    )
                    existing_keys[key] = cur.lastrowid
                    new_ids.append(int(cur.lastrowid))
                    inserted += 1
            conn.commit()

        # Also embed any items missing vectors
        with self._lock:
            have_ids = set(self._vec_ids.tolist())
        with self._connect() as conn:
            all_ids = [r["id"] for r in conn.execute("SELECT id FROM items").fetchall()]
        missing_ids = [i for i in all_ids if i not in have_ids]
        to_embed = sorted(set(new_ids + missing_ids))
        embedded = self._upsert_vectors_for_ids(to_embed)
        return {"inserted": inserted, "updated": updated, "embedded": embedded}

    def sync_browser(self, browser: str) -> dict[str, Any]:
        paths = detect_browser_bookmark_paths().get(browser, [])
        if not paths:
            return {"ok": False, "browser": browser, "error": f"{browser} の Bookmarks が見つかりませんでした。"}
        all_records: list[dict[str, str]] = []
        for p in paths:
            try:
                all_records.extend(parse_chromium_bookmarks_file(p, source=browser))
            except Exception:
                continue
        stats = self.bulk_upsert_items(all_records)
        return {
            "ok": True,
            "browser": browser,
            "profiles_read": len(paths),
            "parsed": len(all_records),
            **stats,
        }

    def sync_html(self, html_text: str) -> dict[str, Any]:
        records = parse_netscape_html_bookmarks(html_text, source="HTML")
        stats = self.bulk_upsert_items(records)
        return {"ok": True, "parsed": len(records), **stats}

    # ---------- Hybrid Search ----------
    def search(
        self,
        query: str = "",
        source: str = "all",
        use_rerank: bool = True,
        limit: int = 40,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        with self._connect() as conn:
            if source and source != "all":
                rows = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM items WHERE source = ? ORDER BY updated_at DESC", (source,)
                    ).fetchall()
                ]
            else:
                rows = [
                    dict(r)
                    for r in conn.execute("SELECT * FROM items ORDER BY updated_at DESC").fetchall()
                ]

        if not rows:
            return {"items": [], "total": 0, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1), "mode": "empty"}

        q = query.strip()
        if not q:
            formatted = [self._format_row(r, score=100) for r in rows[:limit]]
            return {
                "items": formatted,
                "total": len(rows),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                "mode": "recent",
            }

        # 1. Sparse BM25-like scoring
        q_tokens = _tokenize_jp_en(q)
        q_set = set(q_tokens)
        q_low = q.lower()
        doc_tokens_list: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for r in rows:
            text = self._item_embed_text(r)
            toks = _tokenize_jp_en(text)
            c = Counter(toks)
            doc_tokens_list.append(c)
            for t in c.keys():
                if t in q_set:
                    df[t] += 1

        n_docs = len(rows)
        sparse_scores = np.zeros(n_docs, dtype=np.float32)
        for i, (r, c) in enumerate(zip(rows, doc_tokens_list)):
            s = 0.0
            for t in q_tokens:
                tf = c.get(t, 0)
                if tf > 0:
                    idf = math.log((n_docs - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
                    s += idf * (tf * 2.2) / (tf + 1.2)
            # Direct substring bonus (apply mainly for 2+ char queries to avoid 1-char weekday noise like '(金)')
            hay = f"{r['title']} {r['summary']} {r['tags']} {r['folder']}".lower()
            if len(q_low) >= 2 and q_low in hay:
                s += 4.0
            elif len(q_low) == 1 and q_low in hay:
                s += 0.6
            sparse_scores[i] = s

        max_sparse = float(sparse_scores.max()) if n_docs > 0 else 0.0
        sparse_norm = sparse_scores / max_sparse if max_sparse > 0 else sparse_scores

        # 2. Dense Vector scoring (Configured Embedding Server)
        dense_norm = np.zeros(n_docs, dtype=np.float32)
        used_dense = False
        q_vec = self._embed_texts([q])
        if (
            q_vec is not None
            and len(self._vec_ids) > 0
            and self._vec_mat.ndim == 2
            and self._vec_mat.shape[1] == q_vec.shape[1]
        ):
            with self._lock:
                id_to_idx = {int(vid): idx for idx, vid in enumerate(self._vec_ids.tolist())}
                sims_all = (self._vec_mat @ q_vec[0]).astype(np.float32)
            for i, r in enumerate(rows):
                vidx = id_to_idx.get(int(r["id"]))
                if vidx is not None:
                    dense_norm[i] = max(0.0, float(sims_all[vidx]))
            used_dense = True

        # Combine Dense + Sparse (favor dense semantic signal for 1-char ambiguous queries)
        if used_dense:
            w_dense = 0.84 if len(q_low) == 1 else 0.68
            combined = w_dense * dense_norm + (1.0 - w_dense) * sparse_norm
            mode = "hybrid_dense_sparse"
        else:
            combined = sparse_norm
            mode = "sparse_only"

        order = np.argsort(-combined)
        candidates: list[tuple[float, dict[str, Any]]] = []
        for idx in order[: max(limit, 16)]:
            sc = float(combined[idx])
            if used_dense and sc < 0.22 and sparse_scores[idx] <= 0:
                continue
            if not used_dense and sc <= 0:
                continue
            candidates.append((sc, rows[idx]))

        # 3. Optional Local LLM Rerank on Top-10
        if use_rerank and len(candidates) >= 2:
            reranked = self._rerank_with_llm(q, candidates[:10])
            if reranked is not None:
                candidates = reranked + candidates[10:]
                mode += "+llm_rerank"

        out_items = [
            self._format_row(r, score=min(99, max(50, int(round(sc * 100)))))
            for sc, r in candidates[:limit]
        ]
        return {
            "items": out_items,
            "total": len(candidates),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "mode": mode,
        }

    def _rerank_with_llm(
        self, query: str, candidates: list[tuple[float, dict[str, Any]]]
    ) -> list[tuple[float, dict[str, Any]]] | None:
        llm_url = normalize_base_url(self.ai_config.get("llm_url", ""))
        llm_model = (self.ai_config.get("llm_model") or "").strip()
        if not llm_url:
            return None
        lines = []
        for idx, (_, r) in enumerate(candidates, 1):
            lines.append(f"{idx}. {r['title']} | {r['summary'][:80]} | {r['folder']}")
        prompt = (
            "以下の検索クエリに最も関連する候補番号を関連度が高い順にJSON配列のみで返してください。\n"
            f"クエリ: {query}\n"
            "候補:\n" + "\n".join(lines) + "\n"
            "出力形式例: [3, 1, 2]"
        )
        req_body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "関連度順に番号のJSON配列だけを出力してください。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 96,
        }
        if llm_model:
            req_body["model"] = llm_model
        try:
            resp = httpx.post(
                f"{llm_url}/v1/chat/completions",
                json=req_body,
                timeout=8.0,
            )
            resp.raise_for_status()
            content = strip_thinking_tags(resp.json()["choices"][0]["message"]["content"])
            m = re.search(r"\[[\d,\s]+\]", content)
            if not m:
                return None
            order_1based = json.loads(m.group(0))
            seen: set[int] = set()
            reordered: list[tuple[float, dict[str, Any]]] = []
            for rank, num in enumerate(order_1based):
                idx0 = int(num) - 1
                if 0 <= idx0 < len(candidates) and idx0 not in seen:
                    seen.add(idx0)
                    base_sc, row = candidates[idx0]
                    boosted = min(0.99, max(base_sc, 0.96 - rank * 0.03))
                    reordered.append((boosted, row))
            for idx0, pair in enumerate(candidates):
                if idx0 not in seen:
                    reordered.append(pair)
            return reordered
        except Exception:
            return None

    _rerank_with_1918 = _rerank_with_llm

    # ---------- Background AI Enrichment (Phase 2) ----------
    def start_enrichment(self, max_items: int = 20) -> dict[str, Any]:
        if self.enrich_state["running"]:
            return {"started": False, "message": "すでにAI要約エンリッチを実行中です。"}
        t = threading.Thread(target=self._run_enrichment_worker, args=(max_items,), daemon=True)
        t.start()
        return {"started": True, "message": f"最大 {max_items} 件のAI要約エンリッチを開始しました。"}

    def _run_enrichment_worker(self, max_items: int) -> None:
        self.enrich_state = {"running": True, "done": 0, "total": 0, "last_title": "", "error": ""}
        try:
            with self._connect() as conn:
                rows = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM items WHERE enriched = 0 AND url != '' ORDER BY updated_at DESC LIMIT ?",
                        (max_items,),
                    ).fetchall()
                ]
            self.enrich_state["total"] = len(rows)
            updated_ids: list[int] = []

            for r in rows:
                self.enrich_state["last_title"] = r["title"][:40]
                page_meta = self._fetch_page_snippet(r["url"])
                summary, tags = self._summarize_with_1918(r["title"], r["url"], page_meta)
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE items SET summary = ?, tags = ?, enriched = 1, updated_at = ? WHERE id = ?",
                        (summary, json.dumps(tags, ensure_ascii=False), _now_ts(), r["id"]),
                    )
                    conn.commit()
                updated_ids.append(int(r["id"]))
                self.enrich_state["done"] += 1

            if updated_ids:
                self._upsert_vectors_for_ids(updated_ids)
        except Exception as exc:
            self.enrich_state["error"] = str(exc)
        finally:
            self.enrich_state["running"] = False

    def _fetch_page_snippet(self, url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (AreDoko Local SecondBrain)"}
            resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=5.0)
            if resp.status_code >= 400 or "text/html" not in resp.headers.get("content-type", ""):
                return ""
            soup = BeautifulSoup(resp.text[:120000], "html.parser")
            desc = ""
            for attr in ({"name": "description"}, {"property": "og:description"}):
                tag = soup.find("meta", attrs=attr)
                if tag and tag.get("content"):
                    desc = tag["content"].strip()
                    break
            paragraphs = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p")[:5])
            return f"{desc} {paragraphs}".strip()[:600]
        except Exception:
            return ""

    def _summarize_with_llm(self, title: str, url: str, snippet: str) -> tuple[str, list[str]]:
        fallback_summary = snippet[:95] if snippet else f"{_extract_domain(url)} のブックマーク"
        llm_url = normalize_base_url(self.ai_config.get("llm_url", ""))
        llm_model = (self.ai_config.get("llm_model") or "").strip()
        if not llm_url:
            return fallback_summary, []
        prompt = (
            "次のWebページ情報から、曖昧な日本語で後から思い出しやすい「1行要約(50文字以内)」と"
            "「連想キーワード(3〜5個)」をJSONのみで返してください。\n"
            f"タイトル: {title}\nURL: {url}\n抜粋: {snippet[:350]}\n"
            '形式: {"summary": "...", "tags": ["タグ1", "タグ2", "タグ3"]}'
        )
        req_body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "JSONオブジェクトのみを出力してください。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 180,
        }
        if llm_model:
            req_body["model"] = llm_model
        try:
            resp = httpx.post(
                f"{llm_url}/v1/chat/completions",
                json=req_body,
                timeout=15.0,
            )
            resp.raise_for_status()
            content = strip_thinking_tags(resp.json()["choices"][0]["message"]["content"])
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                parsed = json.loads(m.group(0))
                s = str(parsed.get("summary") or fallback_summary).strip()[:120]
                t = [str(x).strip().lstrip("#") for x in parsed.get("tags", []) if str(x).strip()][:5]
                return s, t
        except Exception:
            pass
        return fallback_summary, []

    _summarize_with_1918 = _summarize_with_llm

    # ---------- Status & Formatting ----------
    def get_status(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            enriched = conn.execute("SELECT COUNT(*) FROM items WHERE enriched = 1").fetchone()[0]
            by_source = {
                r["source"]: r["cnt"]
                for r in conn.execute("SELECT source, COUNT(*) AS cnt FROM items GROUP BY source").fetchall()
            }
        browsers = {k: len(v) for k, v in detect_browser_bookmark_paths().items()}
        servers = self.check_servers()
        pct = int(round((enriched / total) * 100)) if total > 0 else 0
        preset_key = self.ai_config.get("preset", "aria_local")
        preset_label = AI_PRESETS.get(preset_key, {}).get("label", preset_key)
        return {
            "total": total,
            "enriched": enriched,
            "enriched_pct": pct,
            "vectors": int(len(self._vec_ids)),
            "vector_dim": int(self._vec_mat.shape[1]) if self._vec_mat.ndim == 2 else 0,
            "by_source": by_source,
            "browsers": browsers,
            "servers": servers,
            "ai_config": dict(self.ai_config),
            "ai_label": preset_label,
            "enrich_state": self.enrich_state,
            "platform": platform.system(),
        }

    def _format_row(self, r: dict[str, Any], score: int = 100) -> dict[str, Any]:
        try:
            tags = json.loads(r.get("tags") or "[]")
        except Exception:
            tags = []
        return {
            "id": r["id"],
            "source": r["source"],
            "title": r["title"],
            "url": r["url"],
            "folder": r["folder"],
            "summary": r["summary"] or (f"{r['folder']} に保存された記憶" if r["url"] else "雑メモ"),
            "tags": tags,
            "domain": r["domain"] or _extract_domain(r["url"]),
            "enriched": bool(r["enriched"]),
            "score": score,
            "updated_at": r["updated_at"],
        }
