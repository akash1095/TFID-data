# Forward Citation Network Pipeline - Complete Guide

## 🎯 Overview

A **production-ready, stop/resume-capable pipeline** for building forward citation knowledge graphs with semantic relationships extracted by LLM.

**What it does:**
1. Fetches **ALL papers** citing "Attention is All You Need" from 2021-2025
2. Builds **citation network** in Neo4j
3. Extracts **semantic relationships** (Extends, Adapts, Analyzes, Outperforms) using Llama 3.1 8B
4. Generates **comprehensive statistics**

**Key Features:**
- ✅ **Stop/Resume** - All steps can be interrupted and resumed
- ✅ **Unlimited Collection** - Fetches ALL papers (not limited to 100)
- ✅ **Async Support** - 4x faster with concurrent LLM requests
- ✅ **Zero-Shot Prompt** - Crisp, focused relationship extraction
- ✅ **Progress Tracking** - Real-time status and statistics

---

## 📁 Files Created

### **Pipeline Scripts**
| File | Purpose | Can Stop/Resume |
|------|---------|-----------------|
| `step1_fetch_citations.py` | Fetch ALL citations from Semantic Scholar | ✅ Yes |
| `step2_extract_relationships.py` | Extract semantic relationships with LLM | ✅ Yes |
| `step3_generate_stats.py` | Generate network statistics | N/A |
| `verify_setup.py` | Verify all prerequisites | N/A |

### **Documentation**
| File | Purpose |
|------|---------|
| `QUICK_START.md` | ⭐ **Start here** - 3-step quick start |
| `PIPELINE_CHECKLIST.md` | Complete pre-flight checklist |
| `README_PIPELINE.md` | This file - complete guide |
| `COMPLETE_PIPELINE_GUIDE.md` | Detailed documentation |
| `CHANGES_SUMMARY.md` | Technical changes to SemanticScholarClient |

### **Modified Core Files**
| File | Changes |
|------|---------|
| `forward_kg_construction/semantic_scholar_client.py` | ✅ Pagination, retry, unlimited collection |
| `forward_kg_construction/llm/prompts.py` | ✅ Zero-shot prompt, 4 relationships only |

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Verify everything is ready
python verify_setup.py

# 2. Fetch ALL citations (2021-2025)
python step1_fetch_citations.py

# 3. Extract relationships (async mode)
python step2_extract_relationships.py --async --max-concurrent 4

# 4. View results
python step3_generate_stats.py
```

---

## 📋 Prerequisites

### 1. Neo4j Database
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password neo4j:latest
```

### 2. Ollama with Llama 3.1 8B
```bash
ollama serve
ollama pull llama3.1:8b
```

### 3. Environment Variables (.env)
```bash
SS_API_KEY=your_api_key_here
NEO4J_PASSWORD=your_password_here
NEO4J_URI=bolt://localhost:7687
```

### 4. Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Pipeline Steps

### **Step 1: Fetch Citations**
```bash
python step1_fetch_citations.py
```

**What it does:**
- Fetches seed paper "Attention is All You Need"
- Fetches **ALL** citing papers from 2021-2025 (unlimited pagination)
- Adds papers to Neo4j with CITES relationships
- Shows year distribution

**Time:** 10-30 minutes (depends on API rate limits)

**Stop/Resume:** ✅ Yes - Already added papers are skipped

**Expected output:**
```
✅ Fetched 2,500 citing papers from 2021:2025

Citations by year:
  2021: 350 papers
  2022: 550 papers
  2023: 700 papers
  2024: 600 papers
  2025: 300 papers

✅ Added 2,500 papers and citation relationships
```

---

### **Step 2: Extract Relationships**

**Async Mode (Recommended - 4x Faster):**
```bash
python step2_extract_relationships.py --async --max-concurrent 4
```

**Sync Mode (Slower but More Stable):**
```bash
python step2_extract_relationships.py
```

**Resume if Interrupted:**
```bash
python step2_extract_relationships.py --resume
```

**What it does:**
- Gets all citation pairs from Neo4j
- Extracts relationships using Llama 3.1 8B:
  - **Extends**: Modifies/improves the method (same domain)
  - **Adapts**: Applies to different domain/task
  - **Analyzes**: Studies the method
  - **Outperforms**: Claims better performance with numbers
- Stores semantic relationships in Neo4j

**Time:** 
- Sync: ~1-2 hours for 1000 pairs
- Async (4 concurrent): ~20-30 minutes for 1000 pairs

**Stop/Resume:** ✅ Yes - Already processed pairs are skipped

**Expected output:**
```
Found 2,500 pairs to process
Extracting relationships...

✅ Processed 2,500 relationships

New relationships extracted:
  EXTENDS: 1,100
  ANALYZES: 650
  ADAPTS: 400
  OUTPERFORMS: 250
  NO_RELATION: 100
```

---

### **Step 3: Generate Statistics**
```bash
python step3_generate_stats.py
```

**What it does:**
- Counts papers, citations, relationships
- Shows distribution by year and type
- Lists top cited papers
- Shows coverage metrics

**Time:** <1 minute

**Expected output:**
```
📊 Total papers in graph: 2,501
🔗 Total CITES relationships: 2,500
🧠 Semantic relationships:
  EXTENDS: 1,100
  ANALYZES: 650
  ADAPTS: 400
  OUTPERFORMS: 250

📈 Semantic relationship coverage: 96.0%
```

---

## 🎛️ Advanced Options

### Limit Number of Papers
Edit `step1_fetch_citations.py` line 95:
```python
max_results=1000  # Limit to 1000 papers instead of unlimited
```

### Use Different LLM Model
```bash
python step2_extract_relationships.py --model llama3.1:70b
```

### Adjust Concurrency
```bash
# More concurrent (faster, more memory)
python step2_extract_relationships.py --async --max-concurrent 8

# Less concurrent (slower, more stable)
python step2_extract_relationships.py --async --max-concurrent 2
```

---

## 🔍 View Results

### Neo4j Browser
Open http://localhost:7474

```cypher
// Citation network
MATCH (citing:Paper)-[:CITES]->(cited:Paper)
WHERE cited.title CONTAINS "Attention"
RETURN citing, cited LIMIT 50

// Semantic relationships
MATCH (citing:Paper)-[r:EXTENDS|ADAPTS|ANALYZES|OUTPERFORMS]->(cited:Paper)
RETURN citing.title, type(r), cited.title, citing.year
ORDER BY citing.year DESC
```

---

## 🛠️ Troubleshooting

Run verification:
```bash
python verify_setup.py
```

Common issues:
- Neo4j not running → `docker start neo4j`
- Ollama not running → `ollama serve`
- Rate limits → Add `SS_API_KEY` to `.env`
- Too slow → Use `--async --max-concurrent 4`

---

## 📚 Documentation

- **QUICK_START.md** - ⭐ Start here
- **PIPELINE_CHECKLIST.md** - Complete checklist
- **COMPLETE_PIPELINE_GUIDE.md** - Full guide

---

**Ready to start?**
```bash
python verify_setup.py
```

