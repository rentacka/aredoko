"""
Verification tests for AreDoko AI Memory (engine + FastAPI endpoints).
Uses an isolated temporary directory so production data/ is never polluted.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import app as aredoko_app
from engine import (
    AreDokoEngine,
    normalize_base_url,
    parse_chromium_bookmarks_file,
    parse_netscape_html_bookmarks,
    strip_thinking_tags,
)


def run_all_tests() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. Chromium JSON parser test
        sample_bm = {
            "roots": {
                "bookmark_bar": {
                    "type": "folder",
                    "name": "ブックマーク バー",
                    "children": [
                        {
                            "type": "url",
                            "name": "Python高速化とNumPyベクトル検索ガイド",
                            "url": "https://example.com/python-numpy",
                        },
                        {
                            "type": "folder",
                            "name": "健康メモ",
                            "children": [
                                {
                                    "type": "url",
                                    "name": "睡眠と自律神経の脳科学エビデンス",
                                    "url": "https://example.com/sleep-brain",
                                }
                            ],
                        },
                    ],
                }
            }
        }
        bm_file = tmp_path / "Bookmarks"
        bm_file.write_text(json.dumps(sample_bm, ensure_ascii=False), encoding="utf-8")
        parsed_chrome = parse_chromium_bookmarks_file(bm_file, source="Chrome")
        assert len(parsed_chrome) == 2, f"Expected 2 chrome items, got {len(parsed_chrome)}"

        # 2. Netscape HTML parser test
        sample_html = """<!DOCTYPE NETSCAPE-Bookmark-file-1>
        <DL><p>
          <DT><H3>投資リサーチ</H3>
          <DL><p>
            <DT><A HREF="https://example.com/pead">決算発表後のドリフトPEAD戦略メモ</A>
          </DL><p>
        </DL><p>"""
        parsed_html = parse_netscape_html_bookmarks(sample_html)
        assert len(parsed_html) == 1 and parsed_html[0]["folder"] == "投資リサーチ"

        # 3. Engine CRUD + Hybrid Search test
        eng = AreDokoEngine(tmp_path / "data")
        stats = eng.bulk_upsert_items(parsed_chrome + parsed_html)
        assert stats["inserted"] == 3, stats
        eng.add_item(title="健康と広告の信頼複利戦略メモ", summary="脱AdSenseで信頼を積む発信戦略", tags=["健康", "広告"])

        res = eng.search("こないだ見た健康と広告の話", use_rerank=False)
        assert len(res["items"]) >= 1, res
        top_title = res["items"][0]["title"]
        assert "健康" in top_title, f"Unexpected top result: {top_title}"
        print(f"[PASS] Hybrid search mode={res['mode']} elapsed={res['elapsed_ms']}ms top='{top_title}'")

        # 4. FastAPI endpoint test (swap engine temporarily)
        orig_engine = aredoko_app.engine
        aredoko_app.engine = eng
        try:
            client = TestClient(aredoko_app.app)
            r_root = client.get("/")
            assert r_root.status_code == 200 and "あれどこだっけ？" in r_root.text

            r_status = client.get("/api/status")
            assert r_status.status_code == 200 and r_status.json()["total"] == 4

            r_search = client.get("/api/search", params={"q": "NumPy 速い", "rerank": "false"})
            assert r_search.status_code == 200 and len(r_search.json()["items"]) >= 1

            r_add = client.post("/api/items", json={"title": "https://x.com/example/status/12345"})
            assert r_add.status_code == 200 and r_add.json()["item"]["source"] == "X"

            item_id = r_add.json()["item"]["id"]
            r_del = client.delete(f"/api/items/{item_id}")
            assert r_del.status_code == 200 and r_del.json()["ok"] is True

            # 5. Multi-provider AI config (Ollama / LM Studio) + namespace isolation + helper tests
            assert normalize_base_url("http://127.0.0.1:11434/v1/embeddings") == "http://127.0.0.1:11434"
            assert normalize_base_url("http://127.0.0.1:1234/v1/") == "http://127.0.0.1:1234"
            assert strip_thinking_tags("<think>reasoning...</think>[2, 1]") == "[2, 1]"

            r_cfg0 = client.get("/api/config")
            assert r_cfg0.status_code == 200 and r_cfg0.json()["vector_file"] == "embeddings.npz"

            r_cfg_ollama = client.post(
                "/api/config",
                json={
                    "preset": "ollama",
                    "embed_url": "http://127.0.0.1:11434/v1/embeddings",
                    "embed_model": "nomic-embed-text",
                    "llm_url": "http://127.0.0.1:11434/v1/chat/completions",
                    "llm_model": "gemma3:4b",
                },
            )
            assert r_cfg_ollama.status_code == 200
            saved = r_cfg_ollama.json()["config"]
            assert saved["preset"] == "ollama"
            assert saved["embed_url"] == "http://127.0.0.1:11434"
            assert saved["embed_model"] == "nomic-embed-text"
            assert saved["llm_model"] == "gemma3:4b"

            r_cfg1 = client.get("/api/config")
            assert r_cfg1.json()["vector_file"].startswith("embeddings_") and r_cfg1.json()["vector_file"] != "embeddings.npz"

            # Verify config persistence across engine restart + dynamic embedding dimension (e.g. 4-dim mock)
            eng_reloaded = AreDokoEngine(tmp_path / "data")
            assert eng_reloaded.ai_config["preset"] == "ollama"
            assert eng_reloaded.ai_config["embed_model"] == "nomic-embed-text"
            import numpy as np

            eng_reloaded._embed_texts = lambda texts: np.tile(np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32), (len(texts), 1))
            reidx = eng_reloaded.reindex_missing_vectors()
            assert reidx["ok"] is True and reidx["embedded"] == 4 and eng_reloaded._vec_mat.shape == (4, 4)

            print("[PASS] All FastAPI endpoints & Ollama/LM Studio multi-provider tests verified")
        finally:
            aredoko_app.engine = orig_engine


if __name__ == "__main__":
    run_all_tests()
    print("ALL TESTS PASSED!")
