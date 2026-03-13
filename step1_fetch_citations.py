#!/usr/bin/env python3
"""
STEP 1: Fetch 2-Level Forward Citation Network from Semantic Scholar (2021-2025)

This script fetches a 2-level forward citation network:
- L1: Papers citing "Attention is All You Need" (2021-2025)
- L2: Papers citing the L1 papers (2021-2025)

The script is resumable - you can stop and restart anytime.

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
MAX_L2_PAPERS_PER_L1 = None  # None = unlimited, or set a number like 100 to limit L2 papers per L1 paper

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


def get_processed_l1_papers(neo4j_client):
    """Get list of L1 papers that already have their L2 citations fetched."""
    with neo4j_client.driver.session() as session:
        # Papers that cite the seed and have outgoing citations (meaning L2 was fetched)
        query = """
        MATCH (l1:Paper)-[:CITES]->(seed:Paper {paper_id: $seed_id})
        OPTIONAL MATCH (l2:Paper)-[:CITES]->(l1)
        WITH l1, count(l2) as l2_count
        WHERE l2_count > 0
        RETURN l1.paper_id as paper_id
        """
        result = session.run(query, seed_id=SEED_PAPER_ID)
        return set(record["paper_id"] for record in result)


def fetch_citations_by_year(ss_client, paper_id, year_filter):
    """Fetch citations year by year (API range filter is broken)."""
    start_year, end_year = map(int, year_filter.split(':'))
    years = range(start_year, end_year + 1)

    all_citations = []
    for year in years:
        year_citations = ss_client.get_paper_citations(
            paper_id=paper_id,
            paginate=True,
            publication_year=str(year),
            max_results=None  # Unlimited
        )
        all_citations.extend(year_citations)

    return all_citations


def main():
    """Fetch 2-level forward citation network."""
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("STEP 1: Fetch 2-Level Forward Citation Network (2021-2025)")
    logger.info("=" * 80)
    logger.info("L1: Papers citing seed paper")
    logger.info("L2: Papers citing L1 papers")
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
        
        # ========================================
        # STEP 1A: Fetch L1 Citations (Seed → L1)
        # ========================================
        logger.info(f"\n{'='*80}")
        logger.info("STEP 1A: Fetching L1 Citations (papers citing seed paper)")
        logger.info(f"{'='*80}")
        logger.info(f"Year range: {YEAR_FILTER}")
        logger.info("Fetching year by year due to API limitations with range filters")
        logger.info("You can stop with Ctrl+C and resume later")

        l1_citations = fetch_citations_by_year(ss_client, SEED_PAPER_ID, YEAR_FILTER)
        logger.info(f"\n✅ Fetched {len(l1_citations)} L1 papers from {YEAR_FILTER}")

        # Show L1 year distribution
        year_counts = {}
        for item in l1_citations:
            citing_paper = item.get('citingPaper', {})
            year = citing_paper.get('year')
            if year:
                year_counts[year] = year_counts.get(year, 0) + 1

        logger.info("\nL1 Citations by year:")
        for year in sorted(year_counts.keys()):
            logger.info(f"  {year}: {year_counts[year]} papers")

        # Add L1 papers to Neo4j
        logger.info(f"\nAdding {len(l1_citations)} L1 papers to Neo4j...")

        l1_paper_ids = []
        for i, item in enumerate(l1_citations, 1):
            citing_paper = item.get('citingPaper', {})
            citing_id = citing_paper.get('paperId')

            if not citing_id:
                continue

            l1_paper_ids.append(citing_id)

            # Add L1 paper (will skip if already exists)
            neo4j_client.add_paper_from_json(citing_paper)

            # Add citation relationship: L1 → Seed
            neo4j_client.add_citation_relationship(
                citing_paper_id=citing_id,
                cited_paper_id=seed_id
            )

            if i % 100 == 0:
                logger.info(f"  Progress: {i}/{len(l1_citations)} L1 papers processed")

        logger.info(f"✅ Added {len(l1_paper_ids)} L1 papers and relationships")

        # ========================================
        # STEP 1B: Fetch L2 Citations (L1 → L2)
        # ========================================
        logger.info(f"\n{'='*80}")
        logger.info("STEP 1B: Fetching L2 Citations (papers citing L1 papers)")
        logger.info(f"{'='*80}")
        logger.info(f"Total L1 papers: {len(l1_paper_ids)}")
        logger.info(f"Year range: {YEAR_FILTER}")

        # Check which L1 papers already have L2 citations
        processed_l1 = get_processed_l1_papers(neo4j_client)
        remaining_l1 = [pid for pid in l1_paper_ids if pid not in processed_l1]

        logger.info(f"Already processed: {len(processed_l1)} L1 papers")
        logger.info(f"Remaining: {len(remaining_l1)} L1 papers")
        logger.info("This may take several hours depending on API rate limits")
        logger.info("You can stop with Ctrl+C and resume later")

        l2_total_count = 0
        l2_year_counts = {}

        for l1_idx, l1_paper_id in enumerate(remaining_l1, 1):
            logger.info(f"\n[{l1_idx}/{len(remaining_l1)}] Fetching L2 for L1 paper: {l1_paper_id}")

            try:
                # Fetch L2 citations for this L1 paper
                l2_citations = fetch_citations_by_year(ss_client, l1_paper_id, YEAR_FILTER)

                # Apply limit if configured
                if MAX_L2_PAPERS_PER_L1 and len(l2_citations) > MAX_L2_PAPERS_PER_L1:
                    logger.info(f"  Limiting to {MAX_L2_PAPERS_PER_L1} L2 papers (found {len(l2_citations)})")
                    l2_citations = l2_citations[:MAX_L2_PAPERS_PER_L1]

                logger.info(f"  Found {len(l2_citations)} L2 papers")

                # Add L2 papers and relationships
                for item in l2_citations:
                    l2_paper = item.get('citingPaper', {})
                    l2_id = l2_paper.get('paperId')

                    if not l2_id:
                        continue

                    # Track year distribution
                    year = l2_paper.get('year')
                    if year:
                        l2_year_counts[year] = l2_year_counts.get(year, 0) + 1

                    # Add L2 paper (will skip if already exists)
                    neo4j_client.add_paper_from_json(l2_paper)

                    # Add citation relationship: L2 → L1
                    neo4j_client.add_citation_relationship(
                        citing_paper_id=l2_id,
                        cited_paper_id=l1_paper_id
                    )

                    l2_total_count += 1

                if l1_idx % 10 == 0:
                    logger.info(f"\n📊 Progress Summary:")
                    logger.info(f"  L1 papers processed: {l1_idx}/{len(remaining_l1)}")
                    logger.info(f"  Total L2 papers added: {l2_total_count}")

            except Exception as e:
                logger.error(f"  Error fetching L2 for {l1_paper_id}: {e}")
                logger.info("  Continuing with next L1 paper...")
                continue

        logger.info(f"\n✅ Added {l2_total_count} L2 papers and relationships")

        if l2_year_counts:
            logger.info("\nL2 Citations by year:")
            for year in sorted(l2_year_counts.keys()):
                logger.info(f"  {year}: {l2_year_counts[year]} papers")
        
        elapsed_time = time.time() - start_time

        logger.info("\n" + "=" * 80)
        logger.info("✅ STEP 1 COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"L1 papers (citing seed): {len(l1_paper_ids)}")
        logger.info(f"L2 papers (citing L1): {l2_total_count}")
        logger.info(f"Total papers in network: {len(l1_paper_ids) + l2_total_count + 1}")  # +1 for seed
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

