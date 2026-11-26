"""Procedural material parameter matcher."""
from .graph import MaterialGraph, REGISTRY, NODE_LABELS
from .engine import fit_material, make_graph_template, edit_target, JobManager
from .search import inverse_search, SearchResult
from .decals import run_witness

__all__ = [
    "MaterialGraph", "REGISTRY", "NODE_LABELS",
    "fit_material", "make_graph_template", "edit_target", "JobManager",
    "inverse_search", "SearchResult", "run_witness",
]


def main():
    """Entry point for ``python3 -m material_matcher`` (web authoring UI)."""
    from .web_app import main as web_main
    web_main()
