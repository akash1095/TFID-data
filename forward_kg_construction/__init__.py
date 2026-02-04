"""
Forward Citation Package

A focused package for forward citation semantic relation extraction and evaluation.
This package provides tools for:
- Fetching forward citations from Semantic Scholar
- Extracting semantic relationships between papers using LLMs
- Storing relationships in Neo4j graph database
- Evaluating extraction quality with instance-level metrics
"""

from forward_kg_construction.semantic_scholar_client import SemanticScholarClient
from forward_kg_construction.settings import Settings

__all__ = [
    "SemanticScholarClient",
    "Settings",
]

