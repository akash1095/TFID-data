#!/usr/bin/env python3
"""
Forward Knowledge Graph Construction CLI

Command-line interface for forward citation semantic relation extraction and evaluation.

Usage:
    # Run extraction with Ollama
    python -m forward_kg_construction extract --model llama3.1:8b --max-concurrent 4

    # Run evaluation (stats mode)
    python -m forward_kg_construction evaluate --mode stats

    # Run automated evaluation with Claude
    python -m forward_kg_construction evaluate --mode automated --sample-size 100
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger


def run_extraction(args):
    """Run forward citation relation extraction."""
    from forward_kg_construction.extractors.paper_relation_extractor import PaperRelationExtractor
    from forward_kg_construction.llm.ollama_inference import (
        OllamaConfig,
        OllamaLLMInference,
        OllamaModel,
    )

    # Get model enum
    try:
        model = OllamaModel(args.model)
    except ValueError:
        logger.error(f"Unknown model: {args.model}")
        logger.info(f"Available models: {[m.value for m in OllamaModel]}")
        sys.exit(1)

    # Initialize LLM client
    llm_config = OllamaConfig(
        model=model,
        base_url=args.ollama_url,
        temperature=0.1,
        num_ctx=8192,  # Set context window to 8K
    )
    llm_client = OllamaLLMInference(config=llm_config)

    # Initialize extractor
    extractor = PaperRelationExtractor(
        uri=os.getenv("NEO4J_URI", args.neo4j_uri),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
        llm_client=llm_client,
        min_delay=args.min_delay,
    )

    try:
        if args.async_mode:
            # Async processing
            logger.info("Running async extraction...")
            results = asyncio.run(
                extractor.process_all_triplets_async(
                    min_citation_count=args.min_citations,
                    head_min_year=args.head_min_year,
                    tail_min_year=args.tail_min_year,
                    max_concurrent=args.max_concurrent,
                    batch_size=args.batch_size,
                )
            )
        else:
            # Sync processing
            logger.info("Running sync extraction...")
            results = extractor.process_all_triplets(
                min_citation_count=args.min_citations,
                head_min_year=args.head_min_year,
                tail_min_year=args.tail_min_year,
            )

        logger.info(f"Extraction complete. Processed {len(results)} triplets with relations.")
    finally:
        extractor.close()


def run_evaluation(args):
    """Run forward citation evaluation."""
    from forward_kg_construction.evaluation.forward_only_evaluation import (
        run_forward_automated_evaluation,
        run_forward_evaluation,
    )

    neo4j_password = os.getenv("NEO4J_PASSWORD")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if args.mode == "stats":
        report = run_forward_evaluation(
            neo4j_uri=args.neo4j_uri,
            neo4j_user="neo4j",
            neo4j_password=neo4j_password,
            evaluation_excel=args.evaluation_excel,
            output_dir=args.output_dir,
            min_citation_count=args.min_citations,
            mode="stats",
        )
        logger.info("Stats evaluation complete.")
        return report

    elif args.mode == "automated":
        if not anthropic_api_key:
            logger.error("ANTHROPIC_API_KEY environment variable required for automated mode")
            sys.exit(1)
        output_file = run_forward_automated_evaluation(
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            neo4j_uri=args.neo4j_uri,
            neo4j_user="neo4j",
            neo4j_password=neo4j_password,
            anthropic_api_key=anthropic_api_key,
            batch_id=args.batch_id,
        )
        logger.info(f"Automated evaluation complete. Output: {output_file}")
        return output_file

    elif args.mode == "metrics":
        if not args.evaluation_excel:
            logger.error("--evaluation-excel required for metrics mode")
            sys.exit(1)
        report = run_forward_evaluation(
            neo4j_uri=args.neo4j_uri,
            neo4j_user="neo4j",
            neo4j_password=neo4j_password,
            evaluation_excel=args.evaluation_excel,
            output_dir=args.output_dir,
            min_citation_count=args.min_citations,
            mode="metrics",
        )
        logger.info("Metrics calculation complete.")
        return report


def main():
    """Main CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Forward Citation Semantic Relation Extraction & Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========================================================================
    # Extract command
    # =========================================================================
    extract_parser = subparsers.add_parser("extract", help="Extract semantic relations")
    extract_parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    extract_parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    extract_parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    extract_parser.add_argument("--min-citations", type=int, default=0, help="Min citation count filter")
    extract_parser.add_argument("--head-min-year", type=int, default=2021, help="Min year for citing paper")
    extract_parser.add_argument("--tail-min-year", type=int, default=2017, help="Min year for cited paper")
    extract_parser.add_argument("--min-delay", type=float, default=0.5, help="Min delay between requests")
    extract_parser.add_argument("--async-mode", action="store_true", help="Use async processing")
    extract_parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent requests (async)")
    extract_parser.add_argument("--batch-size", type=int, default=20, help="Batch size (async)")

    # =========================================================================
    # Evaluate command  
    # =========================================================================
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate extraction quality")
    eval_parser.add_argument("--mode", choices=["stats", "automated", "metrics"], default="stats")
    eval_parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    eval_parser.add_argument("--output-dir", default="./evaluation_output", help="Output directory")
    eval_parser.add_argument("--min-citations", type=int, default=10, help="Min citation count filter")
    eval_parser.add_argument("--sample-size", type=int, default=100, help="Sample size for automated eval")
    eval_parser.add_argument("--evaluation-excel", help="Path to evaluation Excel file")
    eval_parser.add_argument("--batch-id", help="Batch ID to retrieve previous results")

    args = parser.parse_args()

    if args.command == "extract":
        run_extraction(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

