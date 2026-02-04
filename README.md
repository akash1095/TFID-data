# TFID-data

**Forward Citation Knowledge Graph Construction and Semantic Relation Extraction**

This repository contains the academic knowledge graph data and tools for extracting and analyzing semantic relationships between research papers using forward citations.

## Overview

This project builds an academic knowledge graph from Semantic Scholar data and uses Large Language Models (LLMs) to extract semantic relationships between citing and cited papers. The system focuses on four key relationship types:

- **EXTENDS** - Modifies or improves the methodology
- **OUTPERFORMS** - Demonstrates superior performance
- **ADAPTS** - Applies to different domain/task
- **ANALYZES** - Investigates properties/behavior

## Repository Structure

```
TFID-data/
├── tfid-data/
│   └── neo4j.dump                    # Neo4j database dump (~38 MB)
├── forward_kg_construction/          # Main package
│   ├── db_neo4j/                     # Neo4j database operations
│   │   └── academic_graph.py         # Knowledge graph construction
│   ├── llm/                          # LLM inference modules
│   │   ├── llm_inference.py          # Groq LLM client
│   │   ├── ollama_inference.py       # Ollama LLM client
│   │   ├── prompts.py                # Prompt templates
│   │   └── schema.py                 # Pydantic schemas
│   ├── extractors/                   # Relation extraction
│   │   └── paper_relation_extractor.py
│   ├── evaluation/                   # Evaluation scripts
│   │   ├── forward_only_evaluation.py
│   │   └── instance_level_visualization.py
│   ├── logging/                      # Logging utilities
│   ├── semantic_scholar_client.py    # Semantic Scholar API client
│   ├── settings.py                   # Configuration management
│   └── pyproject.toml                # Poetry dependencies
├── NEO4J_SETUP.md                    # Neo4j setup guide
├── .gitignore
└── README.md
```

## Features

- **Knowledge Graph Construction**: Build academic knowledge graphs from Semantic Scholar API
- **LLM-based Relation Extraction**: Extract semantic relationships using Groq or Ollama LLMs
- **Multiple LLM Support**: Compatible with Llama, Mixtral, Gemma, and DeepSeek models
- **Evaluation Framework**: Comprehensive evaluation tools for relationship extraction quality
- **Temporal Analysis**: Track evolution of research relationships over time
- **Visualization**: Generate visualizations for evaluation results

## Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management
- [Neo4j](https://neo4j.com/download/) (version 5.x recommended)
- API Keys:
  - Groq API key (for LLM inference)
  - Semantic Scholar API key (optional, for higher rate limits)

## Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Set Up Neo4j

Follow [NEO4J_SETUP.md](NEO4J_SETUP.md) to install Neo4j and load the database dump.

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_groq_api_key              # For Groq LLM
ANTHROPIC_API_KEY=your_anthropic_api_key    # For evaluation
SS_API_KEY=your_semantic_scholar_api_key    # Optional
```

## Usage

### CLI Commands

#### Extract Semantic Relations

```bash
# Using Ollama (async mode recommended)
python -m forward_kg_construction extract \
    --model llama3.1:8b \
    --async-mode \
    --max-concurrent 4 \
    --head-min-year 2021 \
    --tail-min-year 2017

# Using Groq (sync mode)
python -m forward_kg_construction extract \
    --model llama-3.3-70b-versatile \
    --min-delay 1.0
```

#### Run Evaluation

```bash
# Generate dataset statistics
python -m forward_kg_construction evaluate --mode stats

# Automated evaluation with Claude
python -m forward_kg_construction evaluate \
    --mode automated \
    --sample-size 100

# Calculate metrics from manual evaluation
python -m forward_kg_construction evaluate \
    --mode metrics \
    --evaluation-excel path/to/evaluation.xlsx
```

### Available CLI Options

**Extract Command:**
- `--model`: LLM model name (default: llama3.1:8b)
- `--async-mode`: Enable async processing for faster extraction
- `--max-concurrent`: Max concurrent requests (default: 4)
- `--head-min-year`: Min year for citing papers (default: 2021)
- `--tail-min-year`: Min year for cited papers (default: 2017)
- `--min-citations`: Min citation count filter (default: 0)

**Evaluate Command:**
- `--mode`: Evaluation mode (stats/automated/metrics)
- `--sample-size`: Sample size for automated evaluation (default: 100)
- `--output-dir`: Output directory (default: ./evaluation_output)
- `--evaluation-excel`: Path to evaluation Excel file (for metrics mode)

## Database Schema

**Nodes:**
- `Paper`: Research papers (title, abstract, year, citation_count, etc.)
- `Author`: Paper authors
- `Venue`: Publication venues

**Relationships:**
- `CITES`: Citation relationships
- `EXTENDS/OUTPERFORMS/ADAPTS/ANALYZES`: Semantic relationships (LLM-extracted)
- `AUTHORED_BY`: Paper-author relationships
- `PUBLISHED_IN`: Paper-venue relationships

### Example Queries

```cypher
// Count papers with semantic relationships
MATCH (citing:Paper)-[r:EXTENDS|OUTPERFORMS|ADAPTS|ANALYZES]->(cited:Paper)
RETURN type(r) as RelationType, count(r) as Count

// Find most cited papers
MATCH (p:Paper)
RETURN p.title, p.year, p.citation_count
ORDER BY p.citation_count DESC LIMIT 10
```

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@misc{tfid-data,
  author = {Akash Sujith},
  title = {Forward Citation Knowledge Graph Construction and Semantic Relation Extraction},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/akash1095/TFID-data}
}
```

## License

MIT License

---

**Repository**: [akash1095/TFID-data](https://github.com/akash1095/TFID-data)

