"""
Vercel serverless entry point.
Wraps the FastAPI app so Vercel's @vercel/python builder can serve it.
"""
import sys
from pathlib import Path

# Add backend directory to Python path so imports like
# `from app.services.svg_parser import ...` resolve correctly
backend_dir = str(Path(__file__).resolve().parent.parent / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Import the FastAPI app — Vercel looks for a variable named `app`
from app.main import app
