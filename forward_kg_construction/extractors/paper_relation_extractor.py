import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple

from loguru import logger
from neo4j import GraphDatabase
from pydantic import ValidationError

from forward_kg_construction.llm import LLAMA_8B_EXTRACT_PROMPT, LLAMA_8B_SYSTEM_PROMPT
from forward_kg_construction.llm.llm_inference import LLMInference
from forward_kg_construction.llm.ollama_inference import OllamaLLMInference
from forward_kg_construction.llm.openai_inference import OpenAIInference
from forward_kg_construction.llm.schema import RelationshipAnalysis


class PaperRelationExtractor:
    """
    Extract semantic relations from citation triplets using LLM.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        llm_client: LLMInference | OllamaLLMInference | OpenAIInference,
        min_delay: float = 1,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.llm_client = llm_client
        self.min_delay = min_delay  # seconds between calls

    def close(self):
        """Close database connection."""
        self.driver.close()

    def get_non_processed_triplets(
        self,
        min_citation_count: int = 0,
        head_min_year: int = 2022,
        tail_min_year: int = 2022,
    ) -> List[Dict]:
        """
        Extract citation triplets that have not been processed yet.

        Returns triplets where:
        - head = newer/citing paper
        - tail = older/cited paper
        - head -[:CITES]-> tail
        """
        query = """
        MATCH (head:Paper)-[:CITES]->(tail:Paper)
        WHERE head.abstract IS NOT NULL 
          AND tail.abstract IS NOT NULL 
          AND (tail.citation_count >= $min_citation_count
          OR head.citation_count >= $min_citation_count )
          AND head.year >= $head_min_year 
          AND tail.year >= $tail_min_year
        
          AND NOT EXISTS {
              MATCH (tail)-[r]-(head)
              WHERE type(r) IN [
                'IMPLEMENTS','BUILDS_ON','EXTENDS','ADAPTS',
                'OUTPERFORMS','COMPARES_WITH','CONTRADICTS','ANALYZES',
                'SURVEYS','EVALUATES_ON'
                ]
            }
        
        RETURN tail.paper_id AS tail_id,
               tail.title AS tail_title,
               tail.abstract AS tail_abstract,
               head.paper_id AS head_id,
               head.title AS head_title,
               head.abstract AS head_abstract
        ORDER BY head.citation_count DESC

        """
        with self.driver.session() as session:
            result = session.run(
                query,
                min_citation_count=min_citation_count,
                head_min_year=head_min_year,
                tail_min_year=tail_min_year,
            )
            return [dict(record) for record in result]

    def get_all_triplets(
        self,
        min_citation_count: int = 0,
        head_min_year: int = 2022,
        tail_min_year: int = 2022,
    ) -> List[Dict]:
        """
        Extract all citation triplets with valid abstracts.

        Returns triplets where:
        - head = newer/citing paper
        - tail = older/cited paper
        - head -[:CITES]-> tail
        """
        query = """
        MATCH (head:Paper)-[:CITES]->(tail:Paper)
        WHERE head.abstract IS NOT NULL 
          AND tail.abstract IS NOT NULL 
          AND (tail.citation_count >= $min_citation_count
          OR head.citation_count >= $min_citation_count )
          AND head.year >= $head_min_year 
          AND tail.year >= $tail_min_year
        RETURN tail.paper_id AS tail_id,
               tail.title AS tail_title,
               tail.abstract AS tail_abstract,
               head.paper_id AS head_id,
               head.title AS head_title,
               head.abstract AS head_abstract
        ORDER BY head.citation_count DESC
        """
        with self.driver.session() as session:
            result = session.run(
                query,
                min_citation_count=min_citation_count,
                head_min_year=head_min_year,
                tail_min_year=tail_min_year,
            )
            return [dict(record) for record in result]

    def extract_relation_with_structured_llm(
        self,
        citing_paper: Dict,
        cited_paper: Dict,
        schema: type = RelationshipAnalysis,
        max_retries: int = 2,
    ) -> Optional[RelationshipAnalysis]:
        """
        Extract relation between citing and cited paper using LLM.

        Args:
            citing_paper: The newer paper that cites (contains title and abstract)
            cited_paper: The older paper being cited (contains title and abstract)
            schema: Pydantic schema for structured output
            max_retries: Number of retries for extraction on failure
        """
        prompt = LLAMA_8B_EXTRACT_PROMPT.format(
            citing_title=citing_paper.get("title", "N/A"),
            citing_abstract=citing_paper.get("abstract", "N/A"),
            cited_title=cited_paper.get("title", "N/A"),
            cited_abstract=cited_paper.get("abstract", "N/A"),
        )

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.llm_client.structured_invoke(
                    prompt=prompt, schema=schema, system_prompt=LLAMA_8B_SYSTEM_PROMPT
                )
                if not isinstance(response, RelationshipAnalysis):
                    logger.error(f"{response}")
                return response
            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}")
                if attempt < max_retries:
                    prompt = f"{prompt}\n\nIMPORTANT: Return valid JSON matching the exact schema."
            except Exception as e:
                logger.error(f"LLM extraction failed: {e}")
                return None

        logger.error(f"All retries exhausted: {last_error}")
        return None

    async def _async_extract_single(
        self,
        triplet: Dict,
        semaphore: asyncio.Semaphore,
        schema: type = RelationshipAnalysis,
    ) -> Tuple[Dict, Optional[RelationshipAnalysis]]:
        """
        Async extraction for a single triplet with semaphore-based concurrency control.

        Args:
            triplet: The triplet dict containing paper info
            semaphore: Asyncio semaphore to limit concurrent requests
            schema: Pydantic schema for structured output

        Returns:
            Tuple of (triplet, analysis_result)
        """
        async with semaphore:
            citing_paper = {
                "title": triplet["head_title"],
                "abstract": triplet["head_abstract"],
            }
            cited_paper = {
                "title": triplet["tail_title"],
                "abstract": triplet["tail_abstract"],
            }

            prompt = LLAMA_8B_EXTRACT_PROMPT.format(
                citing_title=citing_paper.get("title", "N/A"),
                citing_abstract=citing_paper.get("abstract", "N/A"),
                cited_title=cited_paper.get("title", "N/A"),
                cited_abstract=cited_paper.get("abstract", "N/A"),
            )

            try:
                response = await self.llm_client.astructured_invoke(
                    prompt=prompt, schema=schema, system_prompt=LLAMA_8B_SYSTEM_PROMPT
                )
                if isinstance(response, RelationshipAnalysis):
                    return (triplet, response)
                else:
                    logger.error(f"Unexpected response type: {type(response)}")
                    return (triplet, None)
            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(f"Extraction failed for {triplet['head_id']}: {e}")
                return (triplet, None)
            except Exception as e:
                logger.error(f"LLM extraction failed for {triplet['head_id']}: {e}")
                return (triplet, None)

    async def process_all_triplets_async(
        self,
        min_citation_count: int = 0,
        head_min_year: int = 2022,
        tail_min_year: int = 2022,
        max_concurrent: int = 4,
        batch_size: int = 20,
    ) -> List[Dict]:
        """
        Process all triplets asynchronously with controlled concurrency.

        This method processes triplets in parallel batches, with a semaphore
        limiting the number of concurrent LLM requests to prevent overloading
        the Ollama instance.

        Args:
            min_citation_count: Minimum citation count filter
            head_min_year: Minimum year for citing paper
            tail_min_year: Minimum year for cited paper
            max_concurrent: Maximum concurrent LLM requests (match OLLAMA_NUM_PARALLEL)
            batch_size: Number of triplets to process before saving results

        Returns:
            List of extraction results
        """
        triplets = self.get_non_processed_triplets(
            min_citation_count, head_min_year, tail_min_year
        )
        logger.info(f"Found {len(triplets)} triplets to process")
        logger.info(
            f"Processing with max_concurrent={max_concurrent}, batch_size={batch_size}"
        )

        # Semaphore limits concurrent requests to Ollama
        semaphore = asyncio.Semaphore(max_concurrent)

        results = []
        total_processed = 0
        total_with_relations = 0

        # Process in batches to provide progress updates and save incrementally
        for batch_start in range(0, len(triplets), batch_size):
            batch_end = min(batch_start + batch_size, len(triplets))
            batch = triplets[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start // batch_size + 1}/"
                f"{(len(triplets) + batch_size - 1) // batch_size} "
                f"(triplets {batch_start + 1}-{batch_end} of {len(triplets)})"
            )

            # Create async tasks for this batch
            tasks = [
                self._async_extract_single(triplet, semaphore) for triplet in batch
            ]

            # Execute batch with gather (preserves order)
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and save to database
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                    continue

                triplet, analysis = result
                total_processed += 1

                if analysis and analysis.relationships:
                    # Filter out No-Relation before counting
                    valid_rels = [
                        r
                        for r in analysis.relationships
                        if r.type.upper().replace("-", "_").replace(" ", "_")
                        != "NO_RELATION"
                    ]

                    if valid_rels:
                        # Save: citing (head) -> relationship -> cited (tail)
                        self.save_relationships(
                            triplet["head_id"], triplet["tail_id"], analysis
                        )
                        results.append(
                            {
                                "citing_id": triplet["head_id"],
                                "cited_id": triplet["tail_id"],
                                "relationships": [
                                    r.model_dump() for r in analysis.relationships
                                ],
                            }
                        )
                        total_with_relations += 1

            logger.info(
                f"Batch complete: {total_processed} processed, "
                f"{total_with_relations} with valid relationships"
            )

        logger.info(
            f"Async processing complete: {total_processed} triplets processed, "
            f"{total_with_relations} with valid relationships"
        )
        return results

    def save_relationships(
        self,
        citing_id: str,
        cited_id: str,
        analysis: RelationshipAnalysis,
    ):
        """
        Save extracted relationships to database without duplication.

        Creates relationships: (citing paper) -[relationship]-> (cited paper)
        where citing paper is the newer paper and cited paper is the older paper.
        """
        if not analysis.relationships:
            return

        for rel in analysis.relationships:
            # Skip "No-Relation" relationships - don't insert them into the database
            if rel.type.upper().replace("-", "_").replace(" ", "_") == "NO_RELATION":
                continue

            relation_label = rel.type.upper().replace("-", "_").replace(" ", "_")
            query = f"""
            MATCH (citing:Paper {{paper_id: $citing_id}})
            MATCH (cited:Paper {{paper_id: $cited_id}})
            MERGE (citing)-[r:{relation_label}]->(cited)
            ON CREATE SET
                r.confidence = $confidence,
                r.evidence = $evidence,
                r.explanation = $explanation,
                r.extracted_by = 'llm',
                r.created_at = datetime()
            ON MATCH SET
                r.confidence = CASE WHEN $confidence = 'high' THEN $confidence ELSE r.confidence END,
                r.updated_at = datetime()
            """
            with self.driver.session() as session:
                session.run(
                    query,
                    citing_id=citing_id,
                    cited_id=cited_id,
                    confidence=rel.confidence,
                    evidence=rel.evidence,
                    explanation=rel.explanation,
                )

    def process_all_triplets(
        self,
        min_citation_count: int = 0,
        head_min_year: int = 2022,
        tail_min_year: int = 2022,
    ) -> List[Dict]:
        """
        Process all triplets and extract semantic relations.

        In the graph: head (newer paper) -[:CITES]-> tail (older paper)
        We extract relationships from citing paper (head) to cited paper (tail).
        """
        triplets = self.get_non_processed_triplets(
            min_citation_count, head_min_year, tail_min_year
        )
        logger.info(f"Found {len(triplets)} triplets to process")

        results = []
        for idx, triplet in enumerate(triplets):
            # In graph: head -[:CITES]-> tail
            # head = newer/citing paper, tail = older/cited paper
            citing_paper = {
                "title": triplet["head_title"],
                "abstract": triplet["head_abstract"],
            }
            cited_paper = {
                "title": triplet["tail_title"],
                "abstract": triplet["tail_abstract"],
            }

            logger.info(
                f"Processing {idx + 1}/{len(triplets)}: "
                f"Citing paper {triplet['head_id']} -> Cited paper {triplet['tail_id']}"
            )

            analysis = self.extract_relation_with_structured_llm(
                citing_paper, cited_paper
            )

            if analysis and analysis.relationships:
                # Save: citing (head) -> relationship -> cited (tail)
                self.save_relationships(
                    triplet["head_id"], triplet["tail_id"], analysis
                )
                results.append(
                    {
                        "citing_id": triplet["head_id"],
                        "cited_id": triplet["tail_id"],
                        "relationships": [
                            r.model_dump() for r in analysis.relationships
                        ],
                    }
                )

            # Rate limiting for Groq API
            time.sleep(self.min_delay)

        logger.info(f"Extracted relationships for {len(results)} triplets")
        return results

    async def process_all_triplets_batch(
        self,
        min_citation_count: int = 0,
        head_min_year: int = 2022,
        tail_min_year: int = 2022,
        batch_size: int = 100,
        max_concurrency: int = 32,
    ) -> List[Dict]:
        """
        Process all triplets using batch processing with OpenAIInference.abatch() (FASTEST!).

        This method uses the abatch() API which is optimized for maximum throughput.
        It processes large batches of requests concurrently, making it ideal for
        OpenAI-compatible endpoints like vLLM, Ollama with OpenAI API, etc.

        Args:
            min_citation_count: Minimum citation count filter
            head_min_year: Minimum year for citing paper
            tail_min_year: Minimum year for cited paper
            batch_size: Number of triplets to process in each batch (default: 100)
            max_concurrency: Maximum concurrent requests (default: 32)

        Returns:
            List of extraction results

        Note:
            This method requires llm_client to be an instance of OpenAIInference.
            It will raise an error if used with other LLM clients.
        """
        # Check if llm_client supports batch processing
        if not isinstance(self.llm_client, OpenAIInference):
            raise ValueError(
                "Batch processing requires OpenAIInference client. "
                f"Current client type: {type(self.llm_client).__name__}"
            )

        triplets = self.get_non_processed_triplets(
            min_citation_count, head_min_year, tail_min_year
        )
        logger.info(f"Found {len(triplets)} triplets to process")
        logger.info(
            f"Batch processing with batch_size={batch_size}, max_concurrency={max_concurrency}"
        )

        results = []
        total_processed = 0
        total_with_relations = 0

        # Process in batches
        for batch_start in range(0, len(triplets), batch_size):
            batch_end = min(batch_start + batch_size, len(triplets))
            batch = triplets[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start // batch_size + 1}/"
                f"{(len(triplets) + batch_size - 1) // batch_size} "
                f"(triplets {batch_start + 1}-{batch_end} of {len(triplets)})"
            )

            # Build messages for all triplets in batch
            messages_list = []
            for idx, triplet in enumerate(batch):
                # Use the full LLAMA_8B_EXTRACT_PROMPT template with instructions
                user_prompt = LLAMA_8B_EXTRACT_PROMPT.format(
                    citing_title=triplet['head_title'],
                    citing_abstract=triplet['head_abstract'],
                    cited_title=triplet['tail_title'],
                    cited_abstract=triplet['tail_abstract']
                )

                # Log first prompt for debugging
                if idx == 0 and batch_start == 0:
                    logger.info(f"Sample prompt (first in batch):")
                    logger.info(f"System: {LLAMA_8B_SYSTEM_PROMPT[:100]}...")
                    logger.info(f"User: {user_prompt[:200]}...")

                messages = self.llm_client.build_messages(
                    system_prompt=LLAMA_8B_SYSTEM_PROMPT,
                    user_prompt=user_prompt
                )
                messages_list.append(messages)

            # Process entire batch with abatch (FASTEST!)
            try:
                batch_results = await self.llm_client.abatch(
                    messages_list,
                    schema=RelationshipAnalysis,
                    max_concurrency=max_concurrency
                )

                # Save results to database
                for triplet, analysis in zip(batch, batch_results):
                    total_processed += 1

                    if analysis and analysis.relationships:
                        # Log what we extracted
                        rel_types = [r.type for r in analysis.relationships]
                        logger.debug(f"Extracted: {rel_types} for {triplet['head_id'][:20]}... -> {triplet['tail_id'][:20]}...")

                        # Filter out No-Relation before counting
                        valid_rels = [
                            r
                            for r in analysis.relationships
                            if r.type.upper().replace("-", "_").replace(" ", "_")
                            != "NO_RELATION"
                        ]

                        if valid_rels:
                            logger.info(f"Valid relationships: {[r.type for r in valid_rels]}")
                            # Save: citing (head) -> relationship -> cited (tail)
                            self.save_relationships(
                                triplet["head_id"], triplet["tail_id"], analysis
                            )
                            results.append(
                                {
                                    "citing_id": triplet["head_id"],
                                    "cited_id": triplet["tail_id"],
                                    "relationships": [
                                        r.model_dump() for r in analysis.relationships
                                    ],
                                }
                            )
                            total_with_relations += 1
                        else:
                            logger.info(f"No valid relationships (all were No-Relation)")

                logger.info(
                    f"Batch complete: {total_processed} processed, "
                    f"{total_with_relations} with valid relationships"
                )

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Continue with next batch instead of failing completely
                continue

        logger.info(
            f"Batch processing complete: {total_processed} triplets processed, "
            f"{total_with_relations} with valid relationships"
        )
        return results
