#!/usr/bin/env python3
"""
STEP 3: Generate Network Statistics

This script generates comprehensive statistics about the citation network
and semantic relationships.

Usage:
    python step3_generate_stats.py
"""

import os
from dotenv import load_dotenv
from loguru import logger
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def main():
    """Generate comprehensive statistics."""
    logger.info("=" * 80)
    logger.info("STEP 3: Network Statistics")
    logger.info("=" * 80)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Total papers
            result = session.run("MATCH (p:Paper) RETURN count(p) as count")
            total_papers = result.single()["count"]
            logger.info(f"\n📊 Total papers in graph: {total_papers}")
            
            # Papers by year
            result = session.run("""
                MATCH (p:Paper)
                WHERE p.year IS NOT NULL
                RETURN p.year as year, count(p) as count
                ORDER BY year DESC
            """)
            logger.info(f"\n📅 Papers by year:")
            for record in result:
                logger.info(f"  {record['year']}: {record['count']} papers")
            
            # Citation relationships
            result = session.run("MATCH ()-[r:CITES]->() RETURN count(r) as count")
            total_cites = result.single()["count"]
            logger.info(f"\n🔗 Total CITES relationships: {total_cites}")
            
            # Semantic relationships
            result = session.run("""
                MATCH ()-[r]->()
                WHERE type(r) IN ['EXTENDS', 'OUTPERFORMS', 'ADAPTS', 'ANALYZES']
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)
            
            logger.info(f"\n🧠 Semantic relationships:")
            total_semantic = 0
            for record in result:
                count = record["count"]
                total_semantic += count
                logger.info(f"  {record['rel_type']}: {count}")
            
            logger.info(f"\n  Total semantic relationships: {total_semantic}")
            
            # NO_RELATION count
            result = session.run("""
                MATCH ()-[r:NO_RELATION]->()
                RETURN count(r) as count
            """)
            no_relation_count = result.single()["count"]
            if no_relation_count > 0:
                logger.info(f"  NO_RELATION: {no_relation_count}")
            
            # Coverage
            if total_cites > 0:
                coverage = (total_semantic / total_cites) * 100
                logger.info(f"\n📈 Semantic relationship coverage: {coverage:.1f}%")
            
            # Top cited papers
            result = session.run("""
                MATCH (p:Paper)
                WHERE p.citation_count IS NOT NULL
                RETURN p.title as title, p.year as year, p.citation_count as citations
                ORDER BY citations DESC
                LIMIT 10
            """)
            
            logger.info(f"\n🏆 Top 10 most cited papers:")
            for i, record in enumerate(result, 1):
                logger.info(f"  {i}. {record['title'][:60]}... ({record['year']}) - {record['citations']} citations")
            
            # Papers with most semantic relationships
            result = session.run("""
                MATCH (p:Paper)-[r]->()
                WHERE type(r) IN ['EXTENDS', 'OUTPERFORMS', 'ADAPTS', 'ANALYZES']
                WITH p, count(r) as rel_count
                ORDER BY rel_count DESC
                LIMIT 10
                RETURN p.title as title, p.year as year, rel_count
            """)
            
            logger.info(f"\n🌟 Papers with most semantic relationships:")
            for i, record in enumerate(result, 1):
                logger.info(f"  {i}. {record['title'][:60]}... ({record['year']}) - {record['rel_count']} relationships")
            
            # Relationship distribution by year
            result = session.run("""
                MATCH (citing:Paper)-[r]->(cited:Paper)
                WHERE type(r) IN ['EXTENDS', 'OUTPERFORMS', 'ADAPTS', 'ANALYZES']
                  AND citing.year >= 2021
                RETURN citing.year as year, type(r) as rel_type, count(r) as count
                ORDER BY year DESC, count DESC
            """)
            
            logger.info(f"\n📊 Relationships by year:")
            current_year = None
            for record in result:
                if record['year'] != current_year:
                    current_year = record['year']
                    logger.info(f"\n  {current_year}:")
                logger.info(f"    {record['rel_type']}: {record['count']}")
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ Statistics generation complete!")
            logger.info("=" * 80)
            logger.info("\nYou can now:")
            logger.info("  1. Query Neo4j Browser: http://localhost:7474")
            logger.info("  2. Run evaluation scripts")
            logger.info("  3. Export data for analysis")
            
    finally:
        driver.close()


if __name__ == "__main__":
    main()

