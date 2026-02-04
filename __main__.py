#!/usr/bin/env python3
"""
Forward Knowledge Graph Construction Package Entry Point

Allows running the package as a module:
    python -m forward_kg_construction extract --model llama3.1:8b
    python -m forward_kg_construction evaluate --mode stats
"""

from cli import main

if __name__ == "__main__":
    main()

