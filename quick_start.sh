#!/bin/bash
# Quick Start Script for Forward Citation Network Pipeline

echo "=========================================="
echo "Forward Citation Network - Quick Start"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating template..."
    cat > .env << EOF
# Semantic Scholar API (optional but recommended)
# Get your key from: https://www.semanticscholar.org/product/api
SS_API_KEY=

# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# Ollama (optional, defaults to localhost)
OLLAMA_BASE_URL=http://localhost:11434
EOF
    echo "✅ Created .env template. Please edit it with your credentials."
    echo ""
fi

# Check if Neo4j is running
echo "Checking Neo4j..."
if nc -z localhost 7687 2>/dev/null; then
    echo "✅ Neo4j is running on port 7687"
else
    echo "❌ Neo4j is not running on port 7687"
    echo ""
    echo "Start Neo4j with Docker:"
    echo "  docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j:latest"
    echo ""
    echo "Or use Neo4j Desktop from: https://neo4j.com/download/"
    exit 1
fi

# Check if Ollama is running
echo "Checking Ollama..."
if nc -z localhost 11434 2>/dev/null; then
    echo "✅ Ollama is running on port 11434"
else
    echo "❌ Ollama is not running on port 11434"
    echo ""
    echo "Install and start Ollama:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo "  ollama serve"
    exit 1
fi

# Check if Llama 3.1 8B is available
echo "Checking Llama 3.1 8B model..."
if ollama list | grep -q "llama3.1:8b"; then
    echo "✅ Llama 3.1 8B model is available"
else
    echo "⚠️  Llama 3.1 8B model not found"
    echo "Pulling model (this may take a few minutes)..."
    ollama pull llama3.1:8b
    if [ $? -eq 0 ]; then
        echo "✅ Model downloaded successfully"
    else
        echo "❌ Failed to download model"
        exit 1
    fi
fi

# Check Python dependencies
echo "Checking Python dependencies..."
if python -c "import requests, neo4j, loguru, langchain_ollama" 2>/dev/null; then
    echo "✅ Python dependencies are installed"
else
    echo "⚠️  Some Python dependencies are missing"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "=========================================="
echo "All checks passed! Ready to run pipeline."
echo "=========================================="
echo ""
echo "Run the pipeline with:"
echo "  python run_pipeline.py"
echo ""
echo "Or test the Semantic Scholar client first:"
echo "  python test_semantic_scholar_client.py"
echo ""

