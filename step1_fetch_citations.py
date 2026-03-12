#!/usr/bin/env python3
"""
STEP 1: Fetch ALL Citations from Semantic Scholar (2021-2025)

This script fetches ALL papers citing "Attention is All You Need" from 2021-2025
and stores them in Neo4j. You can stop and resume this step.

Usage:
    python step1_fetch_citations.py
"""

import os
import time
from dotenv import load_dotenv
from loguru import logger

from forward_kg_construction.semantic_scholar_client import SemanticScholarClient
from forward_kg_construction.db_neo4j.academic_graph import AcademicKnowledgeGraph

# Load environment variables
load_dotenv()

# Configuration
SEED_PAPER_ID = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"  # Attention is All You Need
YEAR_FILTER = "2021:2025"  # Papers from 2021-2025

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Semantic Scholar API
SS_API_KEY = os.getenv("SS_API_KEY")


def check_existing_papers(neo4j_client):
    """Check how many papers already exist in Neo4j."""
    with neo4j_client.driver.session() as session:
        # Count total papers
        result = session.run("MATCH (p:Paper) RETURN count(p) as count")
        total_papers = result.single()["count"]
        
        # Count seed paper
        result = session.run(
            "MATCH (p:Paper {paper_id: $seed_id}) RETURN count(p) as count",
            seed_id=SEED_PAPER_ID
        )
        has_seed = result.single()["count"] > 0
        
        # Count citation relationships
        result = session.run("MATCH ()-[r:CITES]->() RETURN count(r) as count")
        total_cites = result.single()["count"]
        
        return total_papers, has_seed, total_cites


def main():
    """Fetch all citations and build network."""
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("STEP 1: Fetch ALL Citations from Semantic Scholar (2021-2025)")
    logger.info("=" * 80)
    
    # Initialize clients
    logger.info("Initializing Semantic Scholar client...")
    ss_client = SemanticScholarClient(
        api_key=SS_API_KEY,
        rate_limit_delay=3.0,
        max_retries=3
    )
    
    logger.info("Connecting to Neo4j...")
    neo4j_client = AcademicKnowledgeGraph(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    
    try:
        # Check existing data
        total_papers, has_seed, total_cites = check_existing_papers(neo4j_client)
        logger.info(f"\nCurrent Neo4j state:")
        logger.info(f"  Total papers: {total_papers}")
        logger.info(f"  Seed paper exists: {has_seed}")
        logger.info(f"  Total CITES relationships: {total_cites}")
        
        if total_papers > 0:
            logger.warning("\n⚠️  Neo4j already contains papers!")
            response = input("Continue and add more papers? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Aborted by user.")
                return
        
        # Fetch seed paper
        logger.info(f"\nFetching seed paper: {SEED_PAPER_ID}")
        seed_paper = ss_client.get_paper(SEED_PAPER_ID)
        
        if not seed_paper:
            logger.error("Failed to fetch seed paper!")
            return
        
        logger.info(f"✅ Seed paper: {seed_paper.get('title')}")
        logger.info(f"   Year: {seed_paper.get('year')}")
        logger.info(f"   Total citations: {seed_paper.get('citationCount')}")
        
        # Add seed paper if not exists
        if not has_seed:
            logger.info("\nAdding seed paper to Neo4j...")
            seed_id = neo4j_client.add_paper_from_json(seed_paper, return_paper_id=True)
            logger.info(f"✅ Seed paper added: {seed_id}")
        else:
            logger.info("✅ Seed paper already exists in Neo4j")
            seed_id = SEED_PAPER_ID
        
        # Fetch ALL citations (paginated, unlimited)
        logger.info(f"\nFetching ALL citations from {YEAR_FILTER}...")
        logger.info("This will fetch ALL pages - may take 10-30 minutes depending on API rate limits")
        logger.info("You can stop with Ctrl+C and resume later (already added papers will be skipped)")
        
        citations = ss_client.get_paper_citations(
            paper_id=SEED_PAPER_ID,
            paginate=True,  # Fetch all pages
            publication_year=YEAR_FILTER,
            max_results=None  # Unlimited - get ALL papers
        )
        
        logger.info(f"\n✅ Fetched {len(citations)} citing papers from {YEAR_FILTER}")
        
        # Show year distribution
        year_counts = {}
        for item in citations:
            citing_paper = item.get('citingPaper', {})
            year = citing_paper.get('year')
            if year:
                year_counts[year] = year_counts.get(year, 0) + 1
        
        logger.info("\nCitations by year:")
        for year in sorted(year_counts.keys()):
            logger.info(f"  {year}: {year_counts[year]} papers")
        
        # Add papers to Neo4j
        logger.info(f"\nAdding {len(citations)} papers to Neo4j...")
        logger.info("Progress will be saved - you can stop and resume anytime")
        
        added_count = 0
        skipped_count = 0
        
        for i, item in enumerate(citations, 1):
            citing_paper = item.get('citingPaper', {})
            citing_id = citing_paper.get('paperId')
            
            if not citing_id:
                continue
            
            # Add citing paper (will skip if already exists)
            neo4j_client.add_paper_from_json(citing_paper)
            
            # Add citation relationship
            neo4j_client.add_citation_relationship(
                citing_paper_id=citing_id,
                cited_paper_id=seed_id
            )
            
            added_count += 1
            
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(citations)} papers processed")
        
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ STEP 1 COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Total papers fetched: {len(citations)}")
        logger.info(f"Papers added to Neo4j: {added_count}")
        logger.info(f"Execution time: {elapsed_time/60:.2f} minutes")
        logger.info("\nNext step: Run 'python step2_extract_relationships.py'")
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Interrupted by user!")
        logger.info("Progress has been saved to Neo4j.")
        logger.info("You can resume by running this script again.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        neo4j_client.close()


if __name__ == "__main__":
    main()

