"""
AML Neural Network Command Center — Dashboard v2.

Real-time GUI with:
- Live training monitor (SSE streaming)
- Graph visualizer
- Transaction injector
- Alert feed with auto-refresh
- Pipeline trigger
"""

import os
import sys
import json
import sqlite3
import asyncio
import threading
import subprocess
import time
import uuid
import random
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CENTRAL_GRAPH_DB, DASHBOARD_PORT, BANK_DB_PATHS,
    BANK_PORTS, LAUNDERING_PROB_HOLD, LAUNDERING_PROB_REVIEW
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# [Removed dummy alerts insertion block as per request - previously around lines 36-49]
_training_log: list[str] = []
_training_running: bool = False
_pipeline_running: bool = False
_pipeline_last_run: Optional[str] = None
_pipeline_last_alerts_count: int = 0
_alert_feed: list[dict] = []
_injected_tx_count: int = 0


def _account_exists(conn, account_id: Optional[str]) -> Optional[str]:
    if not account_id:
        return None
    row = conn.execute(
        "SELECT account_id FROM kyc WHERE account_id = ?",
        (account_id,)
    ).fetchone()
    return row["account_id"] if row else None


def _score_repeated_small_transfers(
    conn,
    src_account_id: str,
    dst_account_id: str,
    amount: float,
    tx_type: str,
    src_country: str,
    dst_country: str,
    now_iso: str,
) -> tuple[float, dict]:
    """Return a boost for repeated low-value transfers consistent with smurfing."""
    if amount > 1000 or src_country != dst_country or tx_type not in {"internal", "ach", "cash_deposit"}:
        return 0.0, {"pattern": "single-transaction", "recent_small_count": 0, "recent_pair_count": 0}

    rows = conn.execute(
        """
        SELECT amount, timestamp, dst_account_id, tx_type
        FROM transactions
        WHERE src_account_id = ?
        ORDER BY timestamp DESC
        LIMIT 100
        """,
        (src_account_id,),
    ).fetchall()

    now_dt = datetime.fromisoformat(now_iso)
    recent_small_count = 0
    recent_pair_count = 0
    recent_small_total = 0.0

    for row in rows:
        try:
            ts = datetime.fromisoformat(row["timestamp"])
        except (TypeError, ValueError):
            continue

        age_days = max(0.0, (now_dt - ts).total_seconds() / 86400.0)
        if age_days > 14:
            continue

        row_amount = float(row["amount"] or 0.0)
        if row_amount <= 1000 and row["tx_type"] in {"internal", "ach", "cash_deposit"}:
            recent_small_count += 1
            recent_small_total += row_amount
            if row["dst_account_id"] == dst_account_id:
                recent_pair_count += 1

    boost = 0.0
    if recent_small_count >= 5:
        boost += 0.15
    if recent_small_count >= 10:
        boost += 0.20
    if recent_small_count >= 15:
        boost += 0.25
    if recent_pair_count >= 5:
        boost += 0.10
    if recent_small_total >= 3000:
        boost += 0.10
    if recent_small_total >= 5000:
        boost += 0.10

    pattern = "low-and-slow" if boost > 0.0 else "single-transaction"
    return min(boost, 0.6), {
        "pattern": pattern,
        "recent_small_count": recent_small_count,
        "recent_pair_count": recent_pair_count,
        "recent_small_total": round(recent_small_total, 2),
    }

# ─── DB Helpers ───────────────────────────────────────────────────────────────
def get_central_db():
    conn = sqlite3.connect(CENTRAL_GRAPH_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_bank_db(bank_id: str):
    path = BANK_DB_PATHS.get(bank_id)
    if not path or not os.path.exists(path):
        return None
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from bank_node.database import create_central_schema
    create_central_schema(CENTRAL_GRAPH_DB)
    try:
        conn = sqlite3.connect(CENTRAL_GRAPH_DB, check_same_thread=False)
        conn.execute("ALTER TABLE pattern_memory ADD COLUMN source TEXT DEFAULT 'model'")
        conn.commit()
        conn.close()
    except Exception:
        pass  # Column might already exist
    yield

app = FastAPI(title="AML Command Center", lifespan=lifespan)

# Static files
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ─── Training Background Thread ───────────────────────────────────────────────
def _run_training_thread():
    global _training_running, _training_log
    _training_running = True
    _training_log = ["[INFO] Starting training...\n"]

    project_root = os.path.join(BASE_DIR, "..")
    python = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    train_script = os.path.join(project_root, "model", "train.py")

    try:
        proc = subprocess.Popen(
            [python, train_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=project_root
        )
        for line in proc.stdout:
            _training_log.append(line)
        proc.wait()
        _training_log.append(f"[INFO] Training finished (exit code {proc.returncode})\n")
    except Exception as e:
        _training_log.append(f"[ERROR] {e}\n")
    finally:
        _training_running = False

def _run_pipeline_thread():
    global _pipeline_running, _alert_feed
    _pipeline_running = True
    _alert_feed = []

    project_root = os.path.join(BASE_DIR, "..")
    python = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    pipeline_script = os.path.join(project_root, "aggregator", "pipeline.py")

    try:
        proc = subprocess.Popen(
            [python, pipeline_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=project_root
        )
        for line in proc.stdout:
            _training_log.append(f"[PIPELINE] {line}")
        proc.wait()
        _training_log.append(f"[PIPELINE] Done (exit code {proc.returncode})\n")
    except Exception as e:
        _training_log.append(f"[PIPELINE ERROR] {e}\n")
    finally:
        _pipeline_running = False
        _pipeline_last_run = datetime.now().isoformat()
        try:
            conn = get_central_db()
            row = conn.execute("SELECT COUNT(*) as c FROM pattern_memory WHERE source='model'").fetchone()
            _pipeline_last_alerts_count = row["c"] if row else 0
            conn.close()
        except Exception:
            pass

# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    """Live system status."""
    alerts_count = 0
    suspicious_count = 0
    total_accounts = 0
    try:
        for bank_id in ["bank_a", "bank_b", "bank_c"]:
            conn = get_bank_db(bank_id)
            if conn:
                row = conn.execute("SELECT COUNT(*) as c FROM kyc").fetchone()
                total_accounts += row["c"] if row else 0
                row2 = conn.execute("SELECT COUNT(*) as c FROM kyc WHERE is_suspicious=1").fetchone()
                suspicious_count += row2["c"] if row2 else 0
                conn.close()

        conn = get_central_db()
        row = conn.execute("SELECT COUNT(*) as c FROM pattern_memory").fetchone()
        alerts_count = row["c"] if row else 0
        conn.close()
    except Exception:
        pass

    return {
        "training_running": _training_running,
        "pipeline_running": _pipeline_running,
        "total_accounts": total_accounts,
        "suspicious_accounts": suspicious_count,
        "alerts_count": alerts_count,
        "injected_tx_count": _injected_tx_count,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/run-pipeline")
def api_run_pipeline(background_tasks: BackgroundTasks):
    global _pipeline_running
    if _pipeline_running:
        return {"status": "already_running", "error": "Pipeline is currently running"}
    background_tasks.add_task(_run_pipeline_thread)
    return {"status": "started", "alerts_generated": 0, "timestamp": datetime.now().isoformat(), "error": None}

@app.get("/pipeline-status")
def api_pipeline_status():
    global _pipeline_running, _pipeline_last_run, _pipeline_last_alerts_count
    return {
        "running": _pipeline_running,
        "last_run": _pipeline_last_run,
        "last_alert_count": _pipeline_last_alerts_count
    }

@app.get("/api/alerts")
def api_alerts():
    """Get current alerts from central DB."""
    try:
        conn = get_central_db()
        # Fallback order by to support both schemas
        alerts = conn.execute(
            "SELECT * FROM pattern_memory WHERE source = 'model' ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
        conn.close()
        result = []
        for a in alerts:
            d = dict(a)
            try:
                d["account_ids"] = json.loads(d.get("account_ids") or "[]")
                d["countries"] = json.loads(d.get("countries") or "[]")
            except Exception:
                d["account_ids"] = []
                d["countries"] = []
            result.append(d)
        return {"alerts": result}
    except Exception as e:
        return {"alerts": [], "error": str(e)}


@app.get("/api/graph_data")
def api_graph_data():
    """Return node/edge data for graph visualization (sampled for performance)."""
    nodes = []
    edges = []
    try:
        for bank_id in ["bank_a", "bank_b", "bank_c"]:
            conn = get_bank_db(bank_id)
            if not conn:
                continue
            # Sample 200 accounts per bank
            accs = conn.execute(
                "SELECT account_id, is_suspicious FROM kyc ORDER BY RANDOM() LIMIT 200"
            ).fetchall()
            for a in accs:
                nodes.append({
                    "id": a["account_id"],
                    "bank": bank_id,
                    "suspicious": bool(a["is_suspicious"])
                })

            # Get recent transactions
            accs_ids = [a["account_id"] for a in accs]
            if accs_ids:
                placeholders = ",".join("?" * len(accs_ids))
                txs = conn.execute(
                    f"SELECT src_account_id, dst_account_id, amount FROM transactions "
                    f"WHERE src_account_id IN ({placeholders}) LIMIT 500",
                    accs_ids
                ).fetchall()
                for t in txs:
                    edges.append({
                        "source": t["src_account_id"],
                        "target": t["dst_account_id"],
                        "amount": float(t["amount"])
                    })
            conn.close()
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}
    return {"nodes": nodes, "edges": edges}


@app.post("/api/inject_transaction")
async def inject_transaction(
    src_bank: str = Form(...),
    dst_bank: str = Form(...),
    amount: float = Form(...),
    tx_type: str = Form("wire"),
    src_country: str = Form("US"),
    dst_country: str = Form("US"),
    make_suspicious: bool = Form(False),
    src_account_id: Optional[str] = Form(None),
    dst_account_id: Optional[str] = Form(None),
):
    """Inject a synthetic transaction into the live database."""
    global _injected_tx_count, _alert_feed

    try:
        src_conn = get_bank_db(src_bank)
        if not src_conn:
            return JSONResponse({"error": f"Bank {src_bank} DB not available"}, status_code=400)

        # Prefer a locked account pair for repeated testing; otherwise sample one.
        src_acc_id = _account_exists(src_conn, src_account_id)
        if not src_acc_id:
            src_acc = src_conn.execute(
                "SELECT account_id FROM kyc ORDER BY RANDOM() LIMIT 1"
            ).fetchone()
            src_acc_id = src_acc["account_id"] if src_acc else None

        dst_conn = get_bank_db(dst_bank)
        if not dst_conn:
            src_conn.close()
            return JSONResponse({"error": f"Bank {dst_bank} DB not available"}, status_code=400)

        dst_acc_id = _account_exists(dst_conn, dst_account_id)
        if not dst_acc_id:
            dst_acc = dst_conn.execute(
                "SELECT account_id FROM kyc ORDER BY RANDOM() LIMIT 1"
            ).fetchone()
            dst_acc_id = dst_acc["account_id"] if dst_acc else None

        if not src_acc_id or not dst_acc_id:
            return JSONResponse({"error": "Could not find accounts"}, status_code=400)

        tx_id = f"INJECTED-{uuid.uuid4().hex[:12].upper()}"
        now = datetime.now().isoformat()

        src_conn.execute("""
            INSERT OR IGNORE INTO transactions
                (tx_id, src_account_id, dst_account_id, amount, currency,
                 tx_type, timestamp, src_bank_id, dst_bank_id,
                 src_country, dst_country, memo)
            VALUES (?, ?, ?, ?, 'USD', ?, ?, ?, ?, ?, ?, 'GUI Injected')
        """, (
            tx_id, src_acc_id, dst_acc_id,
            amount, tx_type, now,
            src_bank, dst_bank, src_country, dst_country
        ))
        src_conn.commit()
        src_conn.close()
        dst_conn.close()

        _injected_tx_count += 1

        # Compute a mock risk score for the alert feed
        risk = 0.0
        if src_country != dst_country:
            risk += 0.25
        if amount > 5000:
            risk += 0.20
        if amount > 20000:
            risk += 0.30
        if src_bank != dst_bank:
            risk += 0.15
        if make_suspicious:
            risk = min(risk + 0.40, 0.99)

        pattern_boost = 0.0
        pattern_info = {"pattern": "single-transaction", "recent_small_count": 0, "recent_pair_count": 0, "recent_small_total": 0.0}
        if not make_suspicious:
            score_conn = get_bank_db(src_bank)
            if score_conn:
                try:
                    pattern_boost, pattern_info = _score_repeated_small_transfers(
                        score_conn,
                        src_acc_id,
                        dst_acc_id,
                        amount,
                        tx_type,
                        src_country,
                        dst_country,
                        now,
                    )
                finally:
                    score_conn.close()
            risk = min(risk + pattern_boost, 0.99)

        risk = min(risk + random.uniform(0, 0.1), 0.99)

        if risk >= 0.7:
            risk_label = "HIGH"
        elif risk >= 0.4:
            risk_label = "MEDIUM"
        else:
            risk_label = "LOW"

        alert_entry = {
            "tx_id": tx_id,
            "src": f"{src_acc_id} ({src_bank})",
            "dst": f"{dst_acc_id} ({dst_bank})",
            "amount": amount,
            "risk": round(risk, 3),
            "risk_label": risk_label,
            "timestamp": now,
            "src_country": src_country,
            "dst_country": dst_country,
            "src_account_id": src_acc_id,
            "dst_account_id": dst_acc_id,
            "pattern_summary": (
                f"{pattern_info['pattern']} · {pattern_info['recent_small_count']} small txs / 14d"
                if pattern_info["pattern"] == "low-and-slow"
                else "single-transaction"
            ),
        }
        _alert_feed.insert(0, alert_entry)
        _alert_feed = _alert_feed[:50]  # Keep last 50

        return {
            "success": True,
            "tx_id": tx_id,
            "risk_score": round(risk, 3),
            "src_account_id": src_acc_id,
            "dst_account_id": dst_acc_id,
            "pattern_summary": alert_entry["pattern_summary"],
            "alert": alert_entry,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/injected_alerts")
def api_injected_alerts():
    return {"alerts": _alert_feed}


@app.post("/api/start_training")
async def start_training(background_tasks: BackgroundTasks):
    global _training_running, _training_log
    if _training_running:
        return {"status": "already_running"}
    _training_log = []
    background_tasks.add_task(_run_training_thread)
    return {"status": "started"}


@app.post("/api/start_pipeline")
async def start_pipeline(background_tasks: BackgroundTasks):
    global _pipeline_running
    if _pipeline_running:
        return {"status": "already_running"}
    background_tasks.add_task(_run_pipeline_thread)
    return {"status": "started"}


@app.get("/api/training_stream")
async def training_stream(request: Request):
    """Server-Sent Events stream for live training log."""
    async def event_generator():
        last_idx = 0
        while True:
            if await request.is_disconnected():
                break
            current_log = _training_log
            if len(current_log) > last_idx:
                new_lines = current_log[last_idx:]
                for line in new_lines:
                    safe = line.replace("\n", "\\n")
                    yield f"data: {json.dumps({'line': safe, 'running': _training_running})}\n\n"
                last_idx = len(current_log)
            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/bank_stats")
def api_bank_stats():
    """Statistics per bank for the dashboard."""
    stats = {}
    for bank_id in BANK_DB_PATHS.keys():
        conn = get_bank_db(bank_id)
        if not conn:
            stats[bank_id] = {}
            continue
        try:
            total = conn.execute("SELECT COUNT(*) as c FROM kyc").fetchone()["c"]
            suspicious = conn.execute("SELECT COUNT(*) as c FROM kyc WHERE is_suspicious=1").fetchone()["c"]
            tx_count = conn.execute("SELECT COUNT(*) as c FROM transactions").fetchone()["c"]
            stats[bank_id] = {
                "total_accounts": total,
                "suspicious_accounts": suspicious,
                "total_transactions": tx_count,
                "suspicion_rate": round(suspicious / total * 100, 1) if total > 0 else 0
            }
        except Exception as e:
            stats[bank_id] = {"error": str(e)}
        finally:
            conn.close()
    return stats


# ─── Pattern/Trust Score API (v2) ────────────────────────────────────────────

@app.get("/accounts")
def api_accounts(limit: int = 100, bank: Optional[str] = None):
    """
    Returns account-level trust scores, risk scores, and detected patterns.
    Reads from pattern_memory; falls back to DB labels if not scored.
    """
    results = []
    try:
        conn = get_central_db()
        rows = conn.execute(
            "SELECT * FROM pattern_memory WHERE source='model' ORDER BY laundering_prob DESC LIMIT ?",
            (limit,)
        ).fetchall()
        for r in rows:
            d = dict(r)
            results.append({
                "account_id": d.get("hashed_account_id", ""),
                "bank_id": d.get("bank_id", ""),
                "trust_score": round(1.0 - float(d.get("laundering_prob", 0.5)), 4),
                "risk_score": round(float(d.get("laundering_prob", 0.5)), 4),
                "recommendation": d.get("recommendation", "ALLOW"),
                "detected_patterns": json.loads(d.get("detected_patterns") or "[]"),
                "timestamp": d.get("timestamp", ""),
            })
        conn.close()
    except Exception as e:
        return {"accounts": [], "error": str(e)}
    return {"accounts": results, "count": len(results)}


@app.get("/transactions")
def api_transactions(limit: int = 100, bank: Optional[str] = None, min_risk: float = 0.0):
    """
    Returns transaction-level risk scores and pattern labels.
    Reads live transaction rows from bank DBs.
    """
    results = []
    banks_to_query = [bank] if bank else list(BANK_DB_PATHS.keys())
    per_bank = max(1, limit // len(banks_to_query))
    try:
        for bid in banks_to_query:
            conn = get_bank_db(bid)
            if not conn:
                continue
            rows = conn.execute(
                "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ?",
                (per_bank,)
            ).fetchall()
            for r in rows:
                memo = r["memo"] or ""
                detected = []
                for kw in ["chain", "fan_in", "fan_out", "burst", "structuring", "round_trip", "mule_coordination"]:
                    if kw in memo.lower():
                        detected.append(kw)
                near_thresh = 9000 <= float(r["amount"]) < 10000
                risk = 0.7 if detected else (0.5 if near_thresh else 0.1)
                if risk < min_risk:
                    continue
                results.append({
                    "tx_id": r["tx_id"],
                    "from_account": r["src_account_id"],
                    "to_account": r["dst_account_id"],
                    "amount": float(r["amount"]),
                    "timestamp": r["timestamp"],
                    "src_bank": r["src_bank_id"],
                    "dst_bank": r["dst_bank_id"],
                    "src_country": r.get("src_country", ""),
                    "dst_country": r.get("dst_country", ""),
                    "risk_score": risk,
                    "detected_patterns": detected,
                    "near_threshold": near_thresh,
                })
            conn.close()
    except Exception as e:
        return {"transactions": [], "error": str(e)}
    results.sort(key=lambda x: x["risk_score"], reverse=True)
    return {"transactions": results[:limit], "count": len(results[:limit])}


@app.get("/graph")
def api_graph(sample: int = 300):
    """
    Returns annotated graph data: nodes with trust/risk, edges with pattern labels.
    Used by frontend graph visualizer.
    """
    nodes = []
    edges = []
    node_set = set()
    PATTERN_KEYWORDS = ["chain", "fan_in", "fan_out", "burst", "structuring", "round_trip", "mule_coordination"]
    try:
        for bank_id in BANK_DB_PATHS.keys():
            conn = get_bank_db(bank_id)
            if not conn:
                continue
            per_bank = max(1, sample // len(BANK_DB_PATHS))
            accs = conn.execute(
                "SELECT account_id, is_suspicious FROM kyc ORDER BY RANDOM() LIMIT ?",
                (per_bank,)
            ).fetchall()
            for a in accs:
                acc_id = a["account_id"]
                if acc_id not in node_set:
                    node_set.add(acc_id)
                    risk = 0.9 if a["is_suspicious"] else 0.05
                    nodes.append({
                        "id": acc_id,
                        "bank": bank_id,
                        "trust_score": round(1.0 - risk, 2),
                        "risk_score": round(risk, 2),
                        "suspicious": bool(a["is_suspicious"]),
                        "detected_patterns": [],
                    })

            acc_ids = [a["account_id"] for a in accs]
            if acc_ids:
                placeholders = ",".join("?" * len(acc_ids))
                txs = conn.execute(
                    f"SELECT tx_id, src_account_id, dst_account_id, amount, timestamp, memo FROM transactions "
                    f"WHERE src_account_id IN ({placeholders}) LIMIT 1000",
                    acc_ids
                ).fetchall()
                for t in txs:
                    memo = t["memo"] or ""
                    detected = [kw for kw in PATTERN_KEYWORDS if kw in memo.lower()]
                    risk = 0.75 if detected else (0.5 if 9000 <= float(t["amount"]) < 10000 else 0.1)
                    edges.append({
                        "source": t["src_account_id"],
                        "target": t["dst_account_id"],
                        "tx_id": t["tx_id"],
                        "amount": float(t["amount"]),
                        "timestamp": t["timestamp"],
                        "risk_score": risk,
                        "detected_patterns": detected,
                        "pattern_annotation": ", ".join(detected) if detected else "clean",
                    })
            conn.close()
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}
    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


# ─── Main HTML ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE_DIR, "templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=DASHBOARD_PORT, reload=False)
