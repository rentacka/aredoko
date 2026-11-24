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
from engine import AreDokoEngine, parse_chromium_bookmarks_file, parse_netscape_html_bookmarks


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

            print("[PASS] All FastAPI endpoints verified (GET /, /api/status, /api/search, POST/DELETE /api/items)")
        finally:
            aredoko_app.engine = orig_engine


if __name__ == "__main__":
    run_all_tests()
    print("ALL TESTS PASSED!")
