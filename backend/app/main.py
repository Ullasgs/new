"""
NOVA-C — FastAPI Backend
Narrative Output from Visual Analytics – Charts

Routes:
  POST /api/upload          — Upload SVG, get full analysis
  POST /api/analyze         — Re-analyze with different parameters
  POST /api/narrative       — Generate/regenerate narrative
  POST /api/compare         — Multi-chart comparison
  GET  /api/charts          — List all analyzed charts
  GET  /api/charts/{id}     — Get specific chart analysis
  GET  /api/demo            — Load all 7 demo charts
  GET  /api/health          — Health check
"""

from __future__ import annotations
import uuid
import os
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional

from .services.svg_parser import parse_svg
from .services.axis_calibrator import calibrate_chart
from .services.trend_engine import analyze_chart
from .services.llm_narrator import generate_narrative, generate_comparison_narrative
from .services.news_search import search_news, build_search_queries
from .models.schemas import ChartAnalysisResult, ChartInsight, Narrative, NewsEvent
from .routers.auth import router as auth_router, require_auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NOVA-C API",
    description="Narrative Output from Visual Analytics – Charts",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth router
app.include_router(auth_router)

# In-memory store for analyzed charts
chart_store: dict[str, ChartAnalysisResult] = {}
# In-memory store for raw SVG content
svg_store: dict[str, bytes] = {}

# Path to demo SVG files
DEMO_SVG_DIR = Path(__file__).parent.parent.parent / "Chart SVGs"
STATIC_DIR = Path(__file__).parent / "static"


# ─── Request/Response Models ──────────────────────────────────────────────────

class NarrativeRequest(BaseModel):
    chart_id: str
    tone: str = "neutral"  # neutral, bullish, bearish, cautious
    focus_series: str = ""


class CompareRequest(BaseModel):
    chart_ids: list[str]
    tone: str = "neutral"


class CompareResponse(BaseModel):
    narrative: Narrative
    chart_ids: list[str]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "NOVA-C", "charts_loaded": len(chart_store)}


@app.post("/api/upload", response_model=ChartAnalysisResult)
async def upload_chart(file: UploadFile = File(...), user: dict = Depends(require_auth)):
    """Upload an SVG chart, parse it, analyze trends, and generate narrative."""
    if not file.filename or not file.filename.endswith(".svg"):
        raise HTTPException(400, "Only SVG files are accepted")

    content = await file.read()
    chart_id = str(uuid.uuid4())[:8]

    try:
        result = await _process_svg(content, chart_id)
        chart_store[chart_id] = result
        svg_store[chart_id] = content
        return result
    except Exception as e:
        logger.exception(f"Failed to process SVG: {e}")
        raise HTTPException(500, f"Failed to process chart: {str(e)}")


@app.post("/api/narrative", response_model=Narrative)
async def regenerate_narrative(request: NarrativeRequest, user: dict = Depends(require_auth)):
    """Regenerate narrative with different tone or focus."""
    if request.chart_id not in chart_store:
        raise HTTPException(404, f"Chart {request.chart_id} not found")

    result = chart_store[request.chart_id]
    narrative = await generate_narrative(
        result.insight,
        tone=request.tone,
        focus_series=request.focus_series,
    )

    # Update stored result
    result.narrative = narrative
    return narrative


@app.post("/api/compare", response_model=CompareResponse)
async def compare_charts(request: CompareRequest, user: dict = Depends(require_auth)):
    """Compare multiple charts and generate cross-cutting narrative."""
    insights = []
    for cid in request.chart_ids:
        if cid not in chart_store:
            raise HTTPException(404, f"Chart {cid} not found")
        insights.append(chart_store[cid].insight)

    narrative = await generate_comparison_narrative(insights, tone=request.tone)
    return CompareResponse(narrative=narrative, chart_ids=request.chart_ids)


@app.get("/api/charts")
async def list_charts(user: dict = Depends(require_auth)):
    """List all analyzed charts."""
    return [
        {
            "chart_id": cid,
            "title": r.insight.metadata.title,
            "chart_type": r.insight.metadata.chart_type,
            "series_count": len(r.insight.series),
            "confidence": r.insight.overall_confidence,
        }
        for cid, r in chart_store.items()
    ]


@app.get("/api/charts/{chart_id}", response_model=ChartAnalysisResult)
async def get_chart(chart_id: str, user: dict = Depends(require_auth)):
    """Get a specific chart's full analysis."""
    if chart_id not in chart_store:
        raise HTTPException(404, f"Chart {chart_id} not found")
    return chart_store[chart_id]


@app.get("/api/charts/{chart_id}/svg")
async def get_chart_svg(chart_id: str, user: dict = Depends(require_auth)):
    """Return the original raw SVG file for a chart."""
    if chart_id not in svg_store:
        raise HTTPException(404, f"SVG for chart {chart_id} not found")
    return Response(content=svg_store[chart_id], media_type="image/svg+xml")


@app.get("/api/demo")
async def load_demo(user: dict = Depends(require_auth)):
    """Load all 7 demo SVG charts for instant judge engagement."""
    if not DEMO_SVG_DIR.exists():
        raise HTTPException(404, "Demo SVG directory not found")

    results = []
    svg_files = sorted(DEMO_SVG_DIR.glob("*.svg"))

    if not svg_files:
        raise HTTPException(404, "No SVG files found in demo directory")

    for svg_path in svg_files:
        chart_id = str(uuid.uuid4())[:8]
        try:
            content = svg_path.read_bytes()
            result = await _process_svg(content, chart_id)
            chart_store[chart_id] = result
            svg_store[chart_id] = content
            results.append({
                "chart_id": chart_id,
                "title": result.insight.metadata.title,
                "chart_type": result.insight.metadata.chart_type,
                "series_count": len(result.insight.series),
                "confidence": result.insight.overall_confidence,
                "file": svg_path.name,
            })
            logger.info(f"Demo loaded: {svg_path.name} → {chart_id}")
        except Exception as e:
            logger.error(f"Failed to load demo chart {svg_path.name}: {e}")
            results.append({
                "chart_id": chart_id,
                "title": svg_path.name,
                "error": str(e),
            })

    return {"charts": results, "total": len(results)}


# ─── Serve Frontend ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Core Processing Pipeline ─────────────────────────────────────────────────

async def _process_svg(content: bytes, chart_id: str) -> ChartAnalysisResult:
    """Full pipeline: Parse → Calibrate → Analyze → News → Narrate."""

    # Step 1: Parse SVG structure
    chart_data = parse_svg(content)
    chart_data.chart_id = chart_id

    # Step 2: Calibrate pixel coords to real values
    chart_data = calibrate_chart(chart_data)

    # Step 3: Analyze trends, anomalies, correlations
    insight = analyze_chart(chart_data)

    # Step 4: Fetch related news (independent of LLM, always works)
    news_events = await _fetch_news(insight)

    # Step 5: Generate narrative via LLM
    narrative = await generate_narrative(insight)

    return ChartAnalysisResult(insight=insight, narrative=narrative, news_events=news_events)


async def _fetch_news(insight: ChartInsight) -> list[NewsEvent]:
    """Fetch news headlines related to chart trends. Best-effort, never fails."""
    try:
        queries = build_search_queries(
            insight.metadata.title or "",
            [t.model_dump() for t in insight.trends],
            [a.model_dump() for a in insight.anomalies],
        )
        events: list[NewsEvent] = []
        seen_titles = set()
        for q in queries[:4]:
            results = await search_news(q, max_results=3)
            for r in results:
                if r.title not in seen_titles:
                    seen_titles.add(r.title)
                    events.append(NewsEvent(
                        headline=r.title,
                        snippet=r.snippet,
                        url=r.url,
                        date_hint=r.date_hint,
                        search_query=q,
                    ))
        logger.info(f"Fetched {len(events)} news events for '{insight.metadata.title}'")
        return events[:10]
    except Exception as e:
        logger.warning(f"News search failed (non-blocking): {e}")
        return []
