#!/usr/bin/env python3
"""
STEP 2: Extract Semantic Relationships with LLM

This script extracts semantic relationships (Extends, Adapts, Analyzes, Outperforms)
for all citation pairs in Neo4j. You can stop and resume this step.

Usage:
    # Sync mode (slower, more stable)
    python step2_extract_relationships.py

    # Async mode (faster, 4 concurrent requests)
    python step2_extract_relationships.py --async --max-concurrent 4

    # Resume from where you left off
    python step2_extract_relationships.py --resume
"""

import argparse
import asyncio
import os
import time
from dotenv import load_dotenv
from loguru import logger

from forward_kg_construction.extractors.paper_relation_extractor import PaperRelationExtractor
from forward_kg_construction.llm.ollama_inference import OllamaLLMInference, OllamaConfig, OllamaModel
from forward_kg_construction.llm.openai_inference import OpenAIInference, OpenAIConfig

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# LLM Configuration
LLM_MODEL = OllamaModel.LLAMA3_8B
LLM_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def check_extraction_status(extractor):
    """Check how many relationships have been extracted."""
    with extractor.driver.session() as session:
        # Total citation pairs
        result = session.run("""
            MATCH (head:Paper)-[:CITES]->(tail:Paper)
            WHERE head.abstract IS NOT NULL 
              AND tail.abstract IS NOT NULL
              AND head.year >= 2021
              AND tail.year >= 2017
            RETURN count(*) as total
        """)
        total_pairs = result.single()["total"]
        
        # Already processed (have semantic relationships)
        result = session.run("""
            MATCH (head:Paper)-[r]->(tail:Paper)
            WHERE type(r) IN ['EXTENDS', 'OUTPERFORMS', 'ADAPTS', 'ANALYZES', 'NO_RELATION']
              AND head.year >= 2021
              AND tail.year >= 2017
            RETURN count(DISTINCT [head.paper_id, tail.paper_id]) as processed
        """)
        processed_pairs = result.single()["processed"]
        
        # Relationship distribution
        result = session.run("""
            MATCH (head:Paper)-[r]->(tail:Paper)
            WHERE type(r) IN ['EXTENDS', 'OUTPERFORMS', 'ADAPTS', 'ANALYZES', 'NO_RELATION']
              AND head.year >= 2021
              AND tail.year >= 2017
            RETURN type(r) as rel_type, count(r) as count
            ORDER BY count DESC
        """)
        
        rel_counts = {record["rel_type"]: record["count"] for record in result}
        
        return total_pairs, processed_pairs, rel_counts


def main():
    """Extract semantic relationships."""
    parser = argparse.ArgumentParser(description="Extract semantic relationships with LLM")
    parser.add_argument("--async", dest="async_mode", action="store_true", help="Use async mode (faster)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent requests (async mode)")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size (async mode)")
    parser.add_argument("--resume", action="store_true", help="Resume from where you left off")
    parser.add_argument("--model", default="qwen2.5:7b", help="Model name (default: qwen2.5:7b)")
    parser.add_argument("--context-window", type=int, default=8192, help="Context window size (default: 8192)")

    # Backend selection
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                        help="LLM backend to use (default: ollama)")
    parser.add_argument("--openai-url", default="http://localhost:11434/v1",
                        help="OpenAI-compatible API URL (default: http://localhost:11434/v1)")
    parser.add_argument("--openai-api-key", default=None,
                        help="API key for OpenAI-compatible endpoint (e.g., Vast.ai open_button_key)")
    parser.add_argument("--batch-mode", action="store_true",
                        help="Use batch processing mode (requires --backend openai, FASTEST!)")
    parser.add_argument("--batch-concurrency", type=int, default=32,
                        help="Max concurrent requests in batch mode (default: 32)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens for LLM response (default: 1024)")

    args = parser.parse_args()
    
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("STEP 2: Extract Semantic Relationships with LLM")
    logger.info("=" * 80)

    # Validate batch mode
    if args.batch_mode and args.backend != "openai":
        logger.error("--batch-mode requires --backend openai")
        return

    # Initialize LLM client based on backend
    if args.backend == "openai":
        logger.info(f"Initializing OpenAI-compatible LLM ({args.model})...")
        logger.info(f"API URL: {args.openai_url}")
        if args.batch_mode:
            logger.info(f"Batch mode enabled with max_concurrency={args.batch_concurrency}")

        # Get API key from args or environment variable
        api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        if api_key != "EMPTY":
            logger.info(f"Using API key: {api_key[:8]}...")

        llm_config = OpenAIConfig(
            model=args.model,
            temperature=0.1,
            base_url=args.openai_url,
            api_key=api_key,
            max_tokens=args.max_tokens,
        )
        llm_client = OpenAIInference(config=llm_config)
    else:
        logger.info(f"Initializing Ollama LLM ({args.model})...")
        logger.info(f"Context window: {args.context_window} tokens")
        llm_config = OllamaConfig(
            model=OllamaModel(args.model),
            temperature=0.3,
            base_url=LLM_BASE_URL,
            num_ctx=args.context_window  # Set context window
        )
        llm_client = OllamaLLMInference(config=llm_config)
    
    # Initialize extractor
    logger.info("Initializing relation extractor...")
    extractor = PaperRelationExtractor(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        llm_client=llm_client,
        min_delay=0.0
    )
    
    try:
        # Check status
        total_pairs, processed_pairs, rel_counts = check_extraction_status(extractor)
        
        logger.info(f"\nExtraction status:")
        logger.info(f"  Total citation pairs: {total_pairs}")
        logger.info(f"  Already processed: {processed_pairs}")
        logger.info(f"  Remaining: {total_pairs - processed_pairs}")
        
        if rel_counts:
            logger.info(f"\nCurrent relationship distribution:")
            for rel_type, count in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {rel_type}: {count}")
        
        if processed_pairs == total_pairs:
            logger.info("\n✅ All relationships already extracted!")
            logger.info("Run 'python step3_generate_stats.py' to see results")
            return
        
        if processed_pairs > 0 and not args.resume:
            logger.warning(f"\n⚠️  {processed_pairs} relationships already extracted!")
            response = input("Continue and process remaining pairs? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Aborted by user. Use --resume flag to continue.")
                return
        
        # Get unprocessed triplets
        logger.info("\nFetching unprocessed citation pairs...")
        triplets = extractor.get_non_processed_triplets(
            min_citation_count=0,
            head_min_year=2021,
            tail_min_year=2017
        )
        
        logger.info(f"Found {len(triplets)} pairs to process")
        
        if len(triplets) == 0:
            logger.info("✅ No more pairs to process!")
            return
        
        # Process triplets
        logger.info(f"\nExtracting relationships...")

        if args.batch_mode:
            logger.info(f"Mode: Batch (FASTEST!)")
            logger.info(f"Batch size: {args.batch_size}")
            logger.info(f"Max concurrency: {args.batch_concurrency}")
            logger.info("You can stop with Ctrl+C and resume later\n")

            results = asyncio.run(
                extractor.process_all_triplets_batch(
                    min_citation_count=0,
                    head_min_year=2021,
                    tail_min_year=2017,
                    batch_size=args.batch_size,
                    max_concurrency=args.batch_concurrency
                )
            )
        elif args.async_mode:
            logger.info(f"Mode: Async")
            logger.info(f"Max concurrent: {args.max_concurrent}")
            logger.info("You can stop with Ctrl+C and resume later\n")

            results = asyncio.run(
                extractor.process_all_triplets_async(
                    min_citation_count=0,
                    head_min_year=2021,
                    tail_min_year=2017,
                    max_concurrent=args.max_concurrent,
                    batch_size=args.batch_size
                )
            )
        else:
            logger.info(f"Mode: Sync")
            logger.info("You can stop with Ctrl+C and resume later\n")

            results = extractor.process_all_triplets(
                min_citation_count=0,
                head_min_year=2021,
                tail_min_year=2017
            )
        
        # Count results
        rel_counts_new = {}
        for result in results:
            rel_type = result.get('relationship_type', 'NO_RELATION')
            rel_counts_new[rel_type] = rel_counts_new.get(rel_type, 0) + 1
        
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ STEP 2 COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Processed {len(results)} relationships")
        logger.info(f"\nNew relationships extracted:")
        for rel_type, count in sorted(rel_counts_new.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {rel_type}: {count}")
        logger.info(f"\nExecution time: {elapsed_time/60:.2f} minutes")
        logger.info("\nNext step: Run 'python step3_generate_stats.py'")
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Interrupted by user!")
        logger.info("Progress has been saved to Neo4j.")
        logger.info("Resume with: python step2_extract_relationships.py --resume")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        extractor.close()


if __name__ == "__main__":
    main()

