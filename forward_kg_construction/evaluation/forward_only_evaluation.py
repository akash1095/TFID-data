"""
Forward-Only Evaluation for Semantic Relationship Extraction

This module implements evaluation for the forward-only approach focusing on
4 relationship types: Extend, Outperform, Adapt, Analyse.

Sections covered:
- Section 5A: Dataset Statistics
- Section 5B: Temporal Evolution
- Section 7: Manual Evaluation Metrics (reuses automated_evaluation.py)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anthropic import Anthropic
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Forward-only relationship types (4 types)
FORWARD_RELATIONSHIP_TYPES = [
    "EXTENDS",  # Modifies or improves the methodology
    "OUTPERFORMS",  # Demonstrates superior performance
    "ADAPTS",  # Applies to different domain/task
    "ANALYZES",  # Investigates properties/behavior
]

# Mapping for normalization (handles various input formats)
TYPE_NORMALIZATION = {
    # Uppercase
    "EXTENDS": "Extends",
    "OUTPERFORMS": "Outperforms",
    "ADAPTS": "Adapts",
    "ANALYZES": "Analyzes",
    # Title case
    "Extends": "Extends",
    "Outperforms": "Outperforms",
    "Adapts": "Adapts",
    "Analyzes": "Analyzes",
    # Lowercase
    "extends": "Extends",
    "outperforms": "Outperforms",
    "adapts": "Adapts",
    "analyzes": "Analyzes",
    # Variations
    "ADAPTS_FROM": "Adapts",
    "Adapts-from": "Adapts",
    "adapts_from": "Adapts",
    "EXTEND": "Extends",
    "extend": "Extends",
    "Extend": "Extends",
    "OUTPERFORM": "Outperforms",
    "outperform": "Outperforms",
    "Outperform": "Outperforms",
    "ADAPT": "Adapts",
    "adapt": "Adapts",
    "Adapt": "Adapts",
    "ANALYZE": "Analyzes",
    "analyze": "Analyzes",
    "Analyze": "Analyzes",
    "ANALYSE": "Analyzes",
    "analyse": "Analyzes",
    "Analyse": "Analyzes",
    "ANALYSES": "Analyzes",
    "analyses": "Analyzes",
    "Analyses": "Analyzes",
}


@dataclass
class DatasetStatistics:
    """Section 5A: Dataset Statistics"""

    # Total papers in forward citation network
    total_papers: int = 0

    # Temporal distribution (papers per year)
    papers_per_year: Dict[int, int] = field(default_factory=dict)

    # Total relationships extracted (only 4 types)
    total_relationships: int = 0

    # Relationship distribution (count and % for each type)
    relationship_counts: Dict[str, int] = field(default_factory=dict)
    relationship_percentages: Dict[str, float] = field(default_factory=dict)

    # Coverage rate (% of citation edges with semantic relationships)
    total_citation_edges: int = 0
    edges_with_semantic_relations: int = 0
    coverage_rate: float = 0.0


@dataclass
class TemporalEvolution:
    """Section 5B: Temporal Evolution"""

    # Relationships by year (how each type evolved over time)
    relationships_by_year: Dict[int, Dict[str, int]] = field(default_factory=dict)

    # Peak years for each relationship type
    peak_years: Dict[str, int] = field(default_factory=dict)


@dataclass
class ManualEvaluationMetrics:
    """
    Section 7: Manual Evaluation Metrics

    Two-level evaluation:
    1. Instance-level: Agreement distribution + Jaccard similarity
    2. Label-level: Per-type P/R/F1 + micro/macro averaging
    """

    # ================================================================
    # INSTANCE-LEVEL METRICS
    # ================================================================

    # Sample size
    sample_size: int = 0

    # Agreement counts
    n_agree: int = 0
    n_partial: int = 0
    n_disagree: int = 0

    # Agreement rates
    agree_rate: float = 0.0
    partial_rate: float = 0.0
    disagree_rate: float = 0.0

    # Jaccard similarity
    avg_jaccard_score: float = 0.0
    avg_partial_jaccard: float = 0.0  # Average Jaccard for partial matches only
    jaccard_scores: List[float] = field(default_factory=list)

    # Error breakdown (for disagree cases)
    hallucination_count: int = 0  # GT={}, Pred≠{}
    missed_count: int = 0         # GT≠{}, Pred={}
    wrong_types_count: int = 0    # GT≠{}, Pred≠{}, no overlap

    # ================================================================
    # LABEL-LEVEL METRICS
    # ================================================================

    # Per-type metrics: {type: {tp, fp, fn, tn, precision, recall, f1_score, support}}
    per_type_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Micro-averaged metrics
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1_score: float = 0.0

    # Macro-averaged metrics
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1_score: float = 0.0

    # ================================================================
    # LEGACY FIELDS (for backward compatibility)
    # ================================================================
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    misclassification_count: int = 0
    missed_relationship_count: int = 0


class ForwardOnlyEvaluator:
    """
    Evaluator for forward-only semantic relationship extraction.
    Focuses on 4 relationship types: Extend, Outperform, Adapt, Analyse.
    """

    # Seed paper title - "Attention Is All You Need"
    SEED_PAPER_TITLE = "Attention is All you Need"

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        seed_paper_title: str = None,
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.relationship_types = ["Extends", "Outperforms", "Adapts", "Analyzes"]
        self.seed_paper_title = seed_paper_title or self.SEED_PAPER_TITLE

    def close(self):
        """Close database connection."""
        self.driver.close()

    def __del__(self):
        if hasattr(self, "driver"):
            self.driver.close()

    def _normalize_type(self, type_name: str) -> str:
        """Normalize relationship type names to standard format."""
        if not type_name:
            return ""
        type_name = str(type_name).strip()
        # Try direct lookup first
        if type_name in TYPE_NORMALIZATION:
            return TYPE_NORMALIZATION[type_name]
        # Try case-insensitive matching
        type_lower = type_name.lower()
        for key, value in TYPE_NORMALIZATION.items():
            if key.lower() == type_lower:
                return value
        # Return original if no match found
        return type_name

    def _is_forward_type(self, type_name: str) -> bool:
        """Check if relationship type is one of the 4 forward types."""
        normalized = self._normalize_type(type_name)
        return normalized in self.relationship_types

    # =========================================================================
    # Section 5A: Dataset Statistics
    # =========================================================================

    def calculate_dataset_statistics(
        self, min_citation_count: int = 10
    ) -> DatasetStatistics:
        """
        Calculate all dataset statistics (Section 5A).

        Args:
            min_citation_count: Only consider papers with citation_count > this value
                               (seed paper is always included regardless of citation count)
        """
        stats = DatasetStatistics()

        with self.driver.session() as session:
            # Total papers in the network (with citation_count > threshold)
            result = session.run(
                """
                MATCH (p:Paper)
                WHERE p.citation_count > $min_citations
                AND p.year >= 2021 AND p.year <= 2025
                RETURN count(p) as total
                """,
                min_citations=min_citation_count,
            )
            stats.total_papers = result.single()["total"]

            # Papers per year (2021-2025) with citation_count > threshold
            result = session.run(
                """
                MATCH (p:Paper)
                WHERE p.year >= 2021 AND p.year <= 2025
                  AND p.citation_count > $min_citations
                RETURN p.year as year, count(p) as count
                ORDER BY year
                """,
                min_citations=min_citation_count,
            )
            for record in result:
                if record["year"]:
                    stats.papers_per_year[record["year"]] = record["count"]

            # Total citation edges (CITES relationships)
            # p1 (citing): citation_count > threshold OR p2 is seed paper
            # p2 (cited): citation_count > threshold (always required)
            result = session.run(
                """
                MATCH (p1:Paper)-[r:CITES]->(p2:Paper)
                WHERE (p1.citation_count > $min_citations OR p2.title = $seed_title)
                  AND p2.citation_count > $min_citations
                  AND p1.year >= 2021 AND p1.year <= 2025
                RETURN count(r) as total
                """,
                min_citations=min_citation_count,
                seed_title=self.seed_paper_title,
            )
            stats.total_citation_edges = result.single()["total"]

            # Semantic relationships (only 4 forward types)
            # p1 (citing): citation_count > threshold OR p2 is seed paper
            # p2 (cited): citation_count > threshold (always required)
            result = session.run(
                """
                MATCH (p1:Paper)-[r]->(p2:Paper)
                WHERE type(r) IN ['EXTENDS', 'OUTPERFORMS', 'ADAPTS', 'ANALYZES']
                  AND (p1.citation_count > $min_citations OR p2.title = $seed_title)
                  AND p2.citation_count > $min_citations
                  AND p1.year >= 2021 AND p1.year <= 2025
                RETURN type(r) as rel_type, count(r) as count
                """,
                min_citations=min_citation_count,
                seed_title=self.seed_paper_title,
            )
            for record in result:
                rel_type = self._normalize_type(record["rel_type"])
                stats.relationship_counts[rel_type] = record["count"]

            # Calculate totals and percentages
            stats.total_relationships = sum(stats.relationship_counts.values())
            if stats.total_relationships > 0:
                for rel_type, count in stats.relationship_counts.items():
                    stats.relationship_percentages[rel_type] = (
                        count / stats.total_relationships * 100
                    )

            # Edges with semantic relations (unique citing-cited pairs)
            # p1 (citing): citation_count > threshold OR p2 is seed paper
            # p2 (cited): citation_count > threshold (always required)
            result = session.run(
                """
                MATCH (p1:Paper)-[r]->(p2:Paper)
                WHERE type(r) IN ['EXTENDS', 'OUTPERFORMS', 'ADAPTS', 'ANALYZES']
                  AND (p1.citation_count > $min_citations OR p2.title = $seed_title)
                  AND p2.citation_count > $min_citations
                  AND p1.year >= 2021 AND p1.year <= 2025
                RETURN count(DISTINCT [p1.paper_id, p2.paper_id]) as count
                """,
                min_citations=min_citation_count,
                seed_title=self.seed_paper_title,
            )
            stats.edges_with_semantic_relations = result.single()["count"]

            # Coverage rate
            if stats.total_citation_edges > 0:
                stats.coverage_rate = (
                    stats.edges_with_semantic_relations
                    / stats.total_citation_edges
                    * 100
                )

        logger.info(
            f"Dataset Statistics (citation_count > {min_citation_count}): "
            f"{stats.total_papers} papers, {stats.total_relationships} relationships"
        )
        return stats

    # =========================================================================
    # Section 5B: Temporal Evolution
    # =========================================================================

    def calculate_temporal_evolution(
        self, min_citation_count: int = 10
    ) -> TemporalEvolution:
        """
        Calculate temporal evolution of relationships (Section 5B).

        Args:
            min_citation_count: Only consider relationships where cited paper has citation_count > this value
        """
        evolution = TemporalEvolution()

        with self.driver.session() as session:
            # Relationships by year (based on citing paper's year)
            # citing: citation_count > threshold OR cited is seed paper
            # cited: citation_count > threshold (always required)
            result = session.run(
                """
                MATCH (citing:Paper)-[r]->(cited:Paper)
                WHERE type(r) IN ['EXTENDS', 'OUTPERFORMS', 'ADAPTS', 'ANALYZES']
                  AND (citing.citation_count > $min_citations OR cited.title = $seed_title)
                  AND cited.citation_count > $min_citations
                  AND citing.year >= 2021 AND citing.year <= 2025
                RETURN citing.year as year, type(r) as rel_type, count(r) as count
                ORDER BY year, rel_type
                """,
                min_citations=min_citation_count,
                seed_title=self.seed_paper_title,
            )

            # Initialize structure
            for year in range(2021, 2026):
                evolution.relationships_by_year[year] = {
                    rel_type: 0 for rel_type in self.relationship_types
                }

            # Populate counts
            for record in result:
                year = record["year"]
                rel_type = self._normalize_type(record["rel_type"])
                if year and rel_type in self.relationship_types:
                    evolution.relationships_by_year[year][rel_type] = record["count"]

            # Calculate peak years for each type
            for rel_type in self.relationship_types:
                max_count = 0
                peak_year = None
                for year, counts in evolution.relationships_by_year.items():
                    if counts.get(rel_type, 0) > max_count:
                        max_count = counts[rel_type]
                        peak_year = year
                if peak_year:
                    evolution.peak_years[rel_type] = peak_year

        logger.info(
            f"Temporal Evolution (citation_count > {min_citation_count}): "
            f"Peak years = {evolution.peak_years}"
        )
        return evolution

    # =========================================================================
    # Section 7: Manual Evaluation Metrics (Two-Level Evaluation)
    # =========================================================================

    def calculate_manual_evaluation_metrics(
        self,
        evaluation_df: pd.DataFrame,
        system_col: str = "System_Predicted_Types",
        ground_truth_col: str = "Claude_Identified_Types",
        agreement_col: str = "Claude_Agreement",
    ) -> ManualEvaluationMetrics:
        """
        Calculate complete two-level evaluation metrics.

        Level 1 - Instance-Level:
            - Agreement distribution (Agree/Partial/Disagree)
            - Jaccard similarity scores
            - Error breakdown (hallucinations, missed, wrong types)

        Level 2 - Label-Level:
            - Per-type TP/FP/FN/TN counts
            - Per-type Precision/Recall/F1
            - Micro-averaged metrics
            - Macro-averaged metrics

        Args:
            evaluation_df: DataFrame with evaluation results
            system_col: Column name for system predictions
            ground_truth_col: Column name for ground truth labels
            agreement_col: Column name for agreement labels

        Returns:
            ManualEvaluationMetrics with both instance and label level metrics
        """
        metrics = ManualEvaluationMetrics()
        metrics.sample_size = len(evaluation_df)

        if metrics.sample_size == 0:
            return metrics

        # Validate required columns
        required_cols = [system_col, ground_truth_col, agreement_col]
        missing_cols = [c for c in required_cols if c not in evaluation_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.info(f"Available columns: {list(evaluation_df.columns)}")
            return metrics

        # Log column info
        logger.info(f"System predictions: '{system_col}'")
        logger.info(f"Ground truth: '{ground_truth_col}'")
        logger.info(f"Agreement column: '{agreement_col}'")

        # ================================================================
        # INITIALIZE COUNTERS
        # ================================================================

        # Instance-level counters
        n_agree = 0
        n_partial = 0
        n_disagree = 0
        jaccard_scores = []
        partial_jaccard_scores = []  # Jaccard scores for partial matches only

        # Error breakdown
        hallucinations = 0  # GT={}, Pred≠{}
        missed = 0          # GT≠{}, Pred={}
        wrong_types = 0     # GT≠{}, Pred≠{}, no overlap

        # Label-level counters
        type_stats = {
            rel_type: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for rel_type in self.relationship_types
        }

        rows_processed = 0
        rows_skipped = 0

        # ================================================================
        # PROCESS EACH INSTANCE
        # ================================================================
        for idx, row in evaluation_df.iterrows():
            # Get agreement label
            agreement = str(row.get(agreement_col, "")).lower().strip()

            # Skip invalid rows
            if agreement not in ["agree", "partial", "disagree"]:
                if agreement:
                    logger.warning(f"Row {idx}: Unknown agreement '{agreement}', skipping")
                rows_skipped += 1
                continue

            rows_processed += 1

            # Parse types
            pred = self._parse_types(row.get(system_col, ""))
            pred = {self._normalize_type(t) for t in pred}
            pred = {t for t in pred if t in self.relationship_types}

            gt = self._parse_types(row.get(ground_truth_col, ""))
            gt = {self._normalize_type(t) for t in gt}
            gt = {t for t in gt if t in self.relationship_types}

            has_gt = len(gt) > 0
            has_pred = len(pred) > 0

            # Debug first few rows
            if rows_processed <= 3:
                logger.debug(
                    f"Row {idx}: agreement={agreement}, GT={gt}, Pred={pred}"
                )

            # ============================================================
            # INSTANCE-LEVEL: Agreement + Jaccard
            # ============================================================

            # Calculate Jaccard similarity
            if len(gt) == 0 and len(pred) == 0:
                jaccard = 1.0
            elif len(gt.union(pred)) == 0:
                jaccard = 0.0
            else:
                jaccard = len(gt.intersection(pred)) / len(gt.union(pred))

            jaccard_scores.append(jaccard)

            # Count agreement categories
            if agreement == "agree":
                n_agree += 1
            elif agreement == "partial":
                n_partial += 1
                partial_jaccard_scores.append(jaccard)
            elif agreement == "disagree":
                n_disagree += 1

                # Error breakdown for disagree cases
                if not has_gt and has_pred:
                    hallucinations += 1
                elif has_gt and not has_pred:
                    missed += 1
                elif has_gt and has_pred:
                    wrong_types += 1

            # ============================================================
            # LABEL-LEVEL: Per-Type TP/TN/FP/FN
            # ============================================================
            for rel_type in self.relationship_types:
                in_gt = rel_type in gt
                in_pred = rel_type in pred

                if in_gt and in_pred:
                    type_stats[rel_type]["tp"] += 1
                elif in_pred and not in_gt:
                    type_stats[rel_type]["fp"] += 1
                elif in_gt and not in_pred:
                    type_stats[rel_type]["fn"] += 1
                else:  # not in_gt and not in_pred
                    type_stats[rel_type]["tn"] += 1

        # ================================================================
        # POPULATE INSTANCE-LEVEL METRICS
        # ================================================================
        n_total = rows_processed
        metrics.sample_size = n_total

        metrics.n_agree = n_agree
        metrics.n_partial = n_partial
        metrics.n_disagree = n_disagree

        metrics.agree_rate = n_agree / n_total if n_total > 0 else 0
        metrics.partial_rate = n_partial / n_total if n_total > 0 else 0
        metrics.disagree_rate = n_disagree / n_total if n_total > 0 else 0

        metrics.avg_jaccard_score = np.mean(jaccard_scores) if jaccard_scores else 0
        metrics.avg_partial_jaccard = (
            np.mean(partial_jaccard_scores) if partial_jaccard_scores else 0
        )
        metrics.jaccard_scores = jaccard_scores

        metrics.hallucination_count = hallucinations
        metrics.missed_count = missed
        metrics.wrong_types_count = wrong_types

        logger.info(f"Rows processed: {rows_processed}, skipped: {rows_skipped}")
        logger.info(
            f"Instance-Level: Agree={n_agree}, Partial={n_partial}, "
            f"Disagree={n_disagree}, Avg Jaccard={metrics.avg_jaccard_score:.3f}"
        )

        # ================================================================
        # POPULATE LABEL-LEVEL METRICS
        # ================================================================

        precision_list = []
        recall_list = []
        f1_list = []

        for rel_type in self.relationship_types:
            stats = type_stats[rel_type]
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            tn = stats["tn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            support = tp + fn

            metrics.per_type_metrics[rel_type] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        # Micro-averaged metrics
        tp_micro = sum(stats["tp"] for stats in type_stats.values())
        fp_micro = sum(stats["fp"] for stats in type_stats.values())
        fn_micro = sum(stats["fn"] for stats in type_stats.values())

        metrics.micro_precision = (
            tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0
        )
        metrics.micro_recall = (
            tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
        )
        metrics.micro_f1_score = (
            2 * metrics.micro_precision * metrics.micro_recall /
            (metrics.micro_precision + metrics.micro_recall)
            if (metrics.micro_precision + metrics.micro_recall) > 0 else 0
        )

        # Macro-averaged metrics
        metrics.macro_precision = np.mean(precision_list) if precision_list else 0
        metrics.macro_recall = np.mean(recall_list) if recall_list else 0
        metrics.macro_f1_score = np.mean(f1_list) if f1_list else 0

        # Legacy fields (for backward compatibility)
        metrics.precision = metrics.micro_precision
        metrics.recall = metrics.micro_recall
        metrics.f1_score = metrics.micro_f1_score
        metrics.accuracy = (n_agree + n_partial) / n_total if n_total > 0 else 0

        logger.info(
            f"Label-Level: Micro F1={metrics.micro_f1_score:.2%}, "
            f"Macro F1={metrics.macro_f1_score:.2%}"
        )

        return metrics

    def _parse_types(self, types_str: str) -> Set[str]:
        """Parse comma-separated types string into a set."""
        if pd.isna(types_str) or types_str == "" or types_str == "ERROR":
            return set()
        types_str = str(types_str).strip()
        if not types_str:
            return set()
        return {t.strip() for t in types_str.split(",") if t.strip()}

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_full_report(
        self,
        evaluation_df: Optional[pd.DataFrame] = None,
        min_citation_count: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate complete evaluation report with all sections.

        Args:
            evaluation_df: Optional DataFrame with manual evaluation data
            min_citation_count: Only consider papers with citation_count > this value
        """
        report = {
            "filter_criteria": {
                "min_citation_count": min_citation_count,
                "seed_paper_title": self.seed_paper_title,
                "description": f"cited: citation_count > {min_citation_count}; citing: citation_count > {min_citation_count} OR cited is seed paper",
            }
        }

        # Section 5A: Dataset Statistics
        stats = self.calculate_dataset_statistics(min_citation_count=min_citation_count)
        report["dataset_statistics"] = {
            "total_papers": stats.total_papers,
            "papers_per_year": stats.papers_per_year,
            "total_relationships": stats.total_relationships,
            "relationship_counts": stats.relationship_counts,
            "relationship_percentages": stats.relationship_percentages,
            "total_citation_edges": stats.total_citation_edges,
            "edges_with_semantic_relations": stats.edges_with_semantic_relations,
            "coverage_rate": stats.coverage_rate,
        }

        # Section 5B: Temporal Evolution
        evolution = self.calculate_temporal_evolution(
            min_citation_count=min_citation_count
        )
        report["temporal_evolution"] = {
            "relationships_by_year": evolution.relationships_by_year,
            "peak_years": evolution.peak_years,
        }

        # Section 7: Manual Evaluation (if data provided)
        if evaluation_df is not None:
            metrics = self.calculate_manual_evaluation_metrics(evaluation_df)
            report["manual_evaluation"] = {
                "sample_size": metrics.sample_size,
                # Instance-level metrics
                "instance_level": {
                    "n_agree": metrics.n_agree,
                    "n_partial": metrics.n_partial,
                    "n_disagree": metrics.n_disagree,
                    "agree_rate": metrics.agree_rate,
                    "partial_rate": metrics.partial_rate,
                    "disagree_rate": metrics.disagree_rate,
                    "avg_jaccard_score": metrics.avg_jaccard_score,
                    "error_breakdown": {
                        "hallucinations": metrics.hallucination_count,
                        "missed": metrics.missed_count,
                        "wrong_types": metrics.wrong_types_count,
                    },
                },
                # Label-level metrics
                "label_level": {
                    "per_type": metrics.per_type_metrics,
                    "micro": {
                        "precision": metrics.micro_precision,
                        "recall": metrics.micro_recall,
                        "f1_score": metrics.micro_f1_score,
                    },
                    "macro": {
                        "precision": metrics.macro_precision,
                        "recall": metrics.macro_recall,
                        "f1_score": metrics.macro_f1_score,
                    },
                },
                # Legacy fields for backward compatibility
                "overall_metrics": {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                },
                "per_type_metrics": metrics.per_type_metrics,
            }

        return report

    def print_report(self, report: Dict[str, Any]) -> str:
        """Generate formatted text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("FORWARD-ONLY EVALUATION REPORT")
        lines.append("Relationship Types: Extends, Outperforms, Adapts, Analyzes")
        lines.append("=" * 70)

        # Filter criteria
        filter_info = report.get("filter_criteria", {})
        if filter_info:
            min_cit = filter_info.get("min_citation_count", 10)
            seed_title = filter_info.get("seed_paper_title", "")
            lines.append(f"\nFilter: cited paper citation_count > {min_cit}")
            lines.append(f"        citing paper citation_count > {min_cit} OR cited is seed paper")
            lines.append(f"Seed Paper: {seed_title}")
            lines.append("-" * 50)

        # Section 5A
        lines.append("\n## SECTION 5A: DATASET STATISTICS")
        lines.append("-" * 50)
        ds = report.get("dataset_statistics", {})
        lines.append(f"Total Papers:              {ds.get('total_papers', 0):,}")
        lines.append(f"Total Relationships:       {ds.get('total_relationships', 0):,}")
        lines.append(
            f"Total Citation Edges:      {ds.get('total_citation_edges', 0):,}"
        )
        lines.append(f"Coverage Rate:             {ds.get('coverage_rate', 0):.2f}%")

        lines.append("\nPapers per Year:")
        for year in range(2021, 2026):
            count = ds.get("papers_per_year", {}).get(year, 0)
            lines.append(f"  {year}: {count:,}")

        lines.append("\nRelationship Distribution:")
        for rel_type in self.relationship_types:
            count = ds.get("relationship_counts", {}).get(rel_type, 0)
            pct = ds.get("relationship_percentages", {}).get(rel_type, 0)
            lines.append(f"  {rel_type:<12}: {count:>6,} ({pct:>5.1f}%)")

        # Section 5B
        lines.append("\n## SECTION 5B: TEMPORAL EVOLUTION")
        lines.append("-" * 50)
        te = report.get("temporal_evolution", {})

        lines.append("\nRelationships by Year:")
        header = f"{'Year':<6}" + "".join(f"{t:<12}" for t in self.relationship_types)
        lines.append(header)
        lines.append("-" * len(header))
        for year in range(2021, 2026):
            year_data = te.get("relationships_by_year", {}).get(year, {})
            row = f"{year:<6}" + "".join(
                f"{year_data.get(t, 0):<12}" for t in self.relationship_types
            )
            lines.append(row)

        lines.append("\nPeak Years:")
        for rel_type in self.relationship_types:
            peak = te.get("peak_years", {}).get(rel_type, "N/A")
            lines.append(f"  {rel_type:<12}: {peak}")

        # Section 7 (if available)
        if "manual_evaluation" in report:
            lines.append("\n## SECTION 7: MANUAL EVALUATION METRICS")
            lines.append("-" * 60)
            me = report["manual_evaluation"]

            lines.append(f"Sample Size: {me.get('sample_size', 0)}")

            # ============================================================
            # INSTANCE-LEVEL EVALUATION
            # ============================================================
            lines.append("\n" + "=" * 60)
            lines.append("INSTANCE-LEVEL EVALUATION")
            lines.append("=" * 60)

            inst = me.get("instance_level", {})

            lines.append("\nAgreement Distribution:")
            n_agree = inst.get("n_agree", 0)
            n_partial = inst.get("n_partial", 0)
            n_disagree = inst.get("n_disagree", 0)
            agree_rate = inst.get("agree_rate", 0)
            partial_rate = inst.get("partial_rate", 0)
            disagree_rate = inst.get("disagree_rate", 0)

            lines.append(f"  Exact Match (Agree):  {n_agree:>5} ({agree_rate:.1%})")
            lines.append(f"  Partial Match:        {n_partial:>5} ({partial_rate:.1%})")
            lines.append(f"  No Match (Disagree):  {n_disagree:>5} ({disagree_rate:.1%})")

            avg_jaccard = inst.get("avg_jaccard_score", 0)
            lines.append(f"\nAverage Jaccard Score: {avg_jaccard:.3f} ({avg_jaccard:.1%})")

            err = inst.get("error_breakdown", {})
            lines.append("\nError Breakdown (Disagree cases):")
            lines.append(f"  Hallucinations: {err.get('hallucinations', 0)}")
            lines.append(f"  Missed:         {err.get('missed', 0)}")
            lines.append(f"  Wrong Types:    {err.get('wrong_types', 0)}")

            # ============================================================
            # LABEL-LEVEL EVALUATION
            # ============================================================
            lines.append("\n" + "=" * 60)
            lines.append("LABEL-LEVEL EVALUATION")
            lines.append("=" * 60)

            label = me.get("label_level", {})

            lines.append("\nPer-Type Performance:")
            lines.append(f"{'Type':<15} {'P':>8} {'R':>8} {'F1':>8} {'Supp':>6}")
            lines.append("-" * 50)
            for rel_type in self.relationship_types:
                pm = label.get("per_type", {}).get(rel_type, {})
                lines.append(
                    f"{rel_type:<15} {pm.get('precision', 0):>7.1%} "
                    f"{pm.get('recall', 0):>7.1%} {pm.get('f1_score', 0):>7.1%} "
                    f"{pm.get('support', 0):>6}"
                )

            lines.append("\nAggregated Performance:")
            lines.append(f"{'Averaging':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            lines.append("-" * 50)

            micro = label.get("micro", {})
            lines.append(
                f"{'Micro':<15} {micro.get('precision', 0):>9.1%} "
                f"{micro.get('recall', 0):>9.1%} {micro.get('f1_score', 0):>9.1%}"
            )

            macro = label.get("macro", {})
            lines.append(
                f"{'Macro':<15} {macro.get('precision', 0):>9.1%} "
                f"{macro.get('recall', 0):>9.1%} {macro.get('f1_score', 0):>9.1%}"
            )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def export_to_json(self, report: Dict[str, Any], output_path: str) -> None:
        """Export report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report exported to {output_path}")

    def plot_evaluation_metrics(
        self,
        report: Dict[str, Any],
        output_dir: str = "./evaluation_output",
        dpi: int = 300,
    ) -> List[str]:
        """
        Generate academic-quality plots for manual evaluation metrics.

        Uses the new two-level visualization module:
        - Instance-level: Agreement distribution, Jaccard scores
        - Label-level: Per-type P/R/F1, micro/macro comparison

        Args:
            report: Evaluation report dictionary
            output_dir: Directory to save plots
            dpi: Resolution for PNG output (default: 300 for publication quality)

        Returns:
            List of paths to generated plot files
        """
        from forward_kg_construction.evaluation.instance_level_visualization import (
            plot_instance_level_metrics,
            plot_label_level_metrics,
        )

        me = report.get("manual_evaluation", {})
        if not me:
            logger.warning("No manual evaluation data found in report")
            return []

        generated_files = []

        # =====================================================================
        # Instance-Level Plots
        # =====================================================================
        instance_data = me.get("instance_level", {})
        if instance_data:
            # Prepare metrics dict for visualization
            instance_metrics = {
                "n_total": me.get("sample_size", 0),
                "n_agree": instance_data.get("n_agree", 0),
                "n_partial": instance_data.get("n_partial", 0),
                "n_disagree": instance_data.get("n_disagree", 0),
                "agree_rate": instance_data.get("agree_rate", 0),
                "partial_rate": instance_data.get("partial_rate", 0),
                "disagree_rate": instance_data.get("disagree_rate", 0),
                "avg_jaccard_score": instance_data.get("avg_jaccard_score", 0),
                "avg_partial_jaccard": instance_data.get("avg_partial_jaccard", 0),
                "jaccard_scores": instance_data.get("jaccard_scores", []),
            }

            instance_files = plot_instance_level_metrics(
                instance_metrics,
                output_dir=output_dir,
                dpi=dpi,
                prefix="instance_level",
            )
            generated_files.extend(instance_files)

        # =====================================================================
        # Label-Level Plots
        # =====================================================================
        label_data = me.get("label_level", {})
        if label_data:
            label_files = plot_label_level_metrics(
                label_data,
                relationship_types=self.relationship_types,
                output_dir=output_dir,
                dpi=dpi,
                prefix="label_level",
            )
            generated_files.extend(label_files)

        logger.info(f"Generated {len(generated_files)} plots in {output_dir}")
        return generated_files


# =============================================================================
# Forward-Only Automated Evaluator (Reuses automated_evaluation.py logic)
# =============================================================================


class ForwardOnlyAutomatedEvaluator:
    """
    Automated evaluation using Claude for the 4 forward-only relationship types.
    Reuses logic from automated_evaluation.py but focused on:
    Extends, Outperforms, Adapts, Analyzes
    """

    def __init__(
        self,
        anthropic_api_key: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.model = "claude-sonnet-4-20250514"
        self.relationship_types = ["Extends", "Outperforms", "Adapts", "Analyzes"]

    def close(self):
        if hasattr(self, "driver"):
            self.driver.close()

    def __del__(self):
        self.close()

    def get_forward_evaluation_prompt(
        self,
        citing_title: str,
        citing_abstract: str,
        cited_title: str,
        cited_abstract: str,
        system_predicted_types: List[str],
    ) -> str:
        """Generate evaluation prompt focused on 4 forward-only relationship types."""
        prompt = f"""You are an expert scientific literature analyst evaluating semantic relationships between research papers.

**Your Task:** Evaluate the system-predicted relationships using the FORWARD-ONLY taxonomy (4 types only).

---

**CITING PAPER:**
Title: {citing_title}
Abstract: {citing_abstract}

**CITED PAPER:**
Title: {cited_title}
Abstract: {cited_abstract}

---

**FORWARD-ONLY RELATIONSHIP TAXONOMY (4 types):**

1. **Extends**: Modifies or improves the internal structure, algorithm, or methodology of the cited work within the same domain. Keywords: improve, modify, enhance, extend, generalize, variant, build upon.

2. **Outperforms**: Demonstrates empirically superior performance compared to the cited method with quantitative evidence. Requires explicit numerical comparison showing the citing paper's method is better.

3. **Adapts**: Applies the same core method or concept to a different domain, task, or modality. Cross-domain transfer of ideas.

4. **Analyzes**: Investigates, studies, or examines the properties, behavior, limitations, or characteristics of the cited method. The goal is understanding, not building something new.

---

**SYSTEM PREDICTION:**
The automated system identified: {', '.join(system_predicted_types) if system_predicted_types else 'No relationships'}

---

**EVALUATION CRITERIA:**

- Evaluate ONLY against the 4 forward-only types above
- Use SOFT matching - if the prediction is reasonable/plausible, accept it
- "agree" = System prediction is valid
- "partial" = At least one system prediction is valid
- "disagree" = None of the system predictions are valid for these 4 types

---

**OUTPUT FORMAT (JSON only):**

{{
  "relationships": [
    {{
      "type": "Extends|Outperforms|Adapts|Analyzes",
      "evidence": "Specific text from abstracts",
      "confidence": "high|medium|low",
      "justification": "Why this relationship exists"
    }}
  ],
  "no_relationship_reason": "Only if no relationships found",
  "agreement_with_system": "agree|partial|disagree",
  "notes": "Any observations"
}}

**CRITICAL:** Output ONLY valid JSON. No preamble or explanation outside JSON."""
        return prompt

    def sample_forward_edges(self, sample_size: int = 100) -> List[Dict]:
        """Sample citation edges that have forward-only relationship types."""
        with self.driver.session() as session:
            query = """
                MATCH (citing:Paper)-[r]->(cited:Paper)
                WHERE citing.abstract IS NOT NULL
                  AND cited.abstract IS NOT NULL
                  AND citing.year >= 2021 and citing.citation_count>10 and cited.citation_count > 10
                  AND type(r) IN ['EXTENDS', 'OUTPERFORMS', 'ADAPTS', 'ANALYZES']
                  AND r.confidence IN ['high', 'medium']
                WITH citing, cited,
                     collect(type(r)) AS semantic_types,
                     collect({
                         type: type(r),
                         evidence: r.evidence,
                         explanation: r.explanation
                     }) AS relationship_details
                WITH citing, cited, semantic_types, relationship_details, rand() AS random
                ORDER BY random
                LIMIT $limit
                RETURN
                    citing.paper_id AS citing_paper_id,
                    citing.title AS citing_title,
                    citing.abstract AS citing_abstract,
                    citing.year AS citing_year,
                    cited.paper_id AS cited_paper_id,
                    cited.title AS cited_title,
                    cited.abstract AS cited_abstract,
                    cited.year AS cited_year,
                    semantic_types,
                    relationship_details
            """
            edges = list(session.run(query, limit=sample_size))

        logger.info(f"Sampled {len(edges)} forward-only edges for evaluation")

        edges_with_ids = []
        for i, record in enumerate(edges):
            edge_dict = dict(record)
            edge_dict["edge_id"] = i
            edges_with_ids.append(edge_dict)

        return edges_with_ids

    def create_batch_requests(self, edges: List[Dict]) -> List[Dict]:
        """Create batch requests for Claude API."""
        requests = []
        for i, edge in enumerate(edges):
            prompt = self.get_forward_evaluation_prompt(
                citing_title=edge["citing_title"],
                citing_abstract=edge["citing_abstract"],
                cited_title=edge["cited_title"],
                cited_abstract=edge["cited_abstract"],
                system_predicted_types=edge.get("semantic_types", []),
            )
            request = {
                "custom_id": f"edge_{i}",
                "params": {
                    "model": self.model,
                    "max_tokens": 1024,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            requests.append(request)
        return requests

    def process_with_batch_api(
        self, edges: List[Dict], output_dir: str = "./evaluation_batches"
    ) -> str:
        """Process evaluation using Claude's Batch API. Returns batch ID."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        requests = self.create_batch_requests(edges)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        requests_file = Path(output_dir) / f"forward_batch_requests_{timestamp}.jsonl"

        with open(requests_file, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        logger.info(f"Submitting batch with {len(requests)} requests...")
        batch = self.client.messages.batches.create(requests=requests)
        batch_id = batch.id

        # Save metadata
        metadata = {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "num_requests": len(requests),
            "relationship_types": self.relationship_types,
            "edges": edges,
        }
        metadata_file = Path(output_dir) / f"forward_batch_metadata_{batch_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Batch submitted: {batch_id}")
        return batch_id

    def wait_for_batch_completion(
        self, batch_id: str, check_interval: int = 60
    ) -> Dict:
        """Poll batch status until completion."""
        logger.info(f"Waiting for batch {batch_id}...")
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            logger.info(
                f"Status: {status} | "
                f"Succeeded: {batch.request_counts.succeeded}, "
                f"Errored: {batch.request_counts.errored}"
            )

            if status == "ended":
                logger.info("✓ Batch complete!")
                return batch
            elif status in ["failed", "canceled"]:
                logger.error(f"✗ Batch {status}")
                return batch
            time.sleep(check_interval)

    def retrieve_batch_results(
        self, batch_id: str, output_dir: str = "./evaluation_batches"
    ) -> List[Dict]:
        """Retrieve and save batch results."""
        results_response = self.client.messages.batches.results(batch_id)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = (
            Path(output_dir) / f"forward_batch_results_{batch_id}_{timestamp}.jsonl"
        )

        results = []
        with open(results_file, "w") as f:
            for result in results_response:
                f.write(json.dumps(result.model_dump()) + "\n")
                results.append(result.model_dump())

        logger.info(f"Retrieved {len(results)} results")
        return results

    def parse_evaluation_response(self, response_text: str) -> Dict:
        """Parse Claude's JSON response."""
        import re

        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                return {"error": "Failed to parse JSON", "raw_response": response_text}
        return {"error": "No JSON found", "raw_response": response_text}

    def merge_batch_results_with_edges(
        self, edges: List[Dict], batch_results: List[Dict]
    ) -> List[Dict]:
        """Merge batch results with original edge data."""
        results_map = {}
        for result in batch_results:
            custom_id = result.get("custom_id")
            if custom_id:
                if result.get("result", {}).get("type") == "succeeded":
                    message = result["result"]["message"]
                    response_text = message["content"][0]["text"]
                    evaluation = self.parse_evaluation_response(response_text)
                    results_map[custom_id] = {
                        "evaluation": evaluation,
                        "response_raw": response_text,
                    }
                else:
                    error = result.get("result", {}).get("error", {})
                    results_map[custom_id] = {
                        "evaluation": None,
                        "error": error.get("message", "Unknown error"),
                    }

        merged = []
        for edge in edges:
            edge_id = edge["edge_id"]
            custom_id = f"edge_{edge_id}"
            result_data = results_map.get(custom_id, {})
            merged.append(
                {
                    **edge,
                    "claude_evaluation": result_data.get("evaluation"),
                    "claude_response_raw": result_data.get("response_raw"),
                    "error": result_data.get("error"),
                }
            )
        return merged

    def export_to_spreadsheet(
        self, results: List[Dict], output_file: str = "forward_evaluation_results.xlsx"
    ):
        """Export results to Excel for manual verification."""
        rows = []
        for result in results:
            edge_id = result["edge_id"]
            system_types = result.get("semantic_types", [])
            system_types_str = ", ".join(system_types) if system_types else ""

            claude_eval = result.get("claude_evaluation", {})
            if claude_eval and not claude_eval.get("error"):
                claude_relationships = claude_eval.get("relationships", [])
                claude_types = [r["type"] for r in claude_relationships]
                claude_types_str = ", ".join(claude_types)

                claude_details = []
                for rel in claude_relationships:
                    detail = f"{rel['type']}: {rel.get('justification', '')[:100]}"
                    claude_details.append(detail)
                claude_details_str = "\n\n".join(claude_details)

                agreement = claude_eval.get("agreement_with_system", "")
                no_rel_reason = claude_eval.get("no_relationship_reason", "")
                notes = claude_eval.get("notes", "")
            else:
                claude_types_str = "ERROR" if result.get("error") else ""
                claude_details_str = result.get("error", "")
                agreement = ""
                no_rel_reason = ""
                notes = ""

            row = {
                "Edge_ID": edge_id,
                "Citing_Paper": result["citing_title"],
                "Citing_Year": result.get("citing_year", ""),
                "Cited_Paper": result["cited_title"],
                "Cited_Year": result.get("cited_year", ""),
                "System_Predicted_Types": system_types_str,
                "Claude_Identified_Types": claude_types_str,
                "Claude_Agreement": agreement,
                "Claude_Details": claude_details_str,
                "No_Relationship_Reason": no_rel_reason,
                "Claude_Notes": notes,
                "Ground_Truth_Types": "",  # For manual annotation
                "Human_Notes": "",
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Forward_Evaluation", index=False)
            workbook = writer.book
            worksheet = writer.sheets["Forward_Evaluation"]

            header_format = workbook.add_format(
                {
                    "bold": True,
                    "bg_color": "#4472C4",
                    "font_color": "white",
                    "border": 1,
                }
            )
            wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})

            worksheet.set_column("A:A", 10)
            worksheet.set_column("B:B", 40, wrap_format)
            worksheet.set_column("C:C", 10)
            worksheet.set_column("D:D", 40, wrap_format)
            worksheet.set_column("E:E", 10)
            worksheet.set_column("F:F", 25, wrap_format)
            worksheet.set_column("G:G", 25, wrap_format)
            worksheet.set_column("H:H", 15)
            worksheet.set_column("I:I", 50, wrap_format)
            worksheet.set_column("J:J", 30, wrap_format)
            worksheet.set_column("K:K", 30, wrap_format)
            worksheet.set_column("L:L", 25, wrap_format)
            worksheet.set_column("M:M", 30, wrap_format)

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            worksheet.freeze_panes(1, 0)

        logger.info(f"✓ Exported {len(rows)} rows to {output_file}")
        return output_file


def run_forward_automated_evaluation(
    sample_size: int = 100,
    output_dir: str = "./evaluation_output",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = None,
    anthropic_api_key: str = None,
    batch_id: str = None,
) -> str:
    """
    Run forward-only automated evaluation using Claude batch API.

    Args:
        sample_size: Number of edges to sample (ignored if batch_id is provided)
        output_dir: Directory to save output files
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        anthropic_api_key: Anthropic API key
        batch_id: Optional batch ID to retrieve results from a previous batch
                  (skips sampling and submission if provided)

    Returns:
        Path to the exported Excel file
    """
    evaluator = ForwardOnlyAutomatedEvaluator(
        anthropic_api_key=anthropic_api_key,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )

    try:
        if batch_id:
            # Skip sampling and submission - retrieve from previous batch
            logger.info(f"Using existing batch: {batch_id}")

            # Load edges from saved metadata file
            metadata_file = Path(output_dir) / f"forward_batch_metadata_{batch_id}.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                edges = metadata.get("edges", [])
                logger.info(f"Loaded {len(edges)} edges from metadata file")
            else:
                logger.warning(f"Metadata file not found: {metadata_file}")
                logger.warning("Will retrieve results without edge data")
                edges = []

            # Step 1: Wait for completion (in case still processing)
            logger.info("Step 1: Checking batch completion...")
            evaluator.wait_for_batch_completion(batch_id, check_interval=30)

            # Step 2: Retrieve results
            logger.info("Step 2: Retrieving batch results...")
            batch_results = evaluator.retrieve_batch_results(
                batch_id, output_dir=output_dir
            )

            # Step 3: Merge results
            logger.info("Step 3: Merging results...")
            merged_results = evaluator.merge_batch_results_with_edges(edges, batch_results)

            # Step 4: Export to Excel
            logger.info("Step 4: Exporting to spreadsheet...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{output_dir}/forward_batch_evaluation_{batch_id}_{timestamp}.xlsx"
            evaluator.export_to_spreadsheet(merged_results, output_file=output_file)

        else:
            # Full flow: sample, submit, wait, retrieve
            # Step 1: Sample edges
            logger.info("Step 1: Sampling forward-only edges...")
            edges = evaluator.sample_forward_edges(sample_size=sample_size)

            # Step 2: Submit batch
            logger.info("Step 2: Submitting batch to Claude API...")
            batch_id = evaluator.process_with_batch_api(edges, output_dir=output_dir)

            # Step 3: Wait for completion
            logger.info("Step 3: Waiting for batch completion...")
            evaluator.wait_for_batch_completion(batch_id, check_interval=60)

            # Step 4: Retrieve results
            logger.info("Step 4: Retrieving batch results...")
            batch_results = evaluator.retrieve_batch_results(
                batch_id, output_dir=output_dir
            )

            # Step 5: Merge results
            logger.info("Step 5: Merging results...")
            merged_results = evaluator.merge_batch_results_with_edges(edges, batch_results)

            # Step 6: Export to Excel - batch/automated mode
            logger.info("Step 6: Exporting to spreadsheet...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{output_dir}/forward_batch_evaluation_{timestamp}.xlsx"
            evaluator.export_to_spreadsheet(merged_results, output_file=output_file)

        logger.info(f"\n{'='*60}")
        logger.info("✓ Forward-only evaluation complete!")
        logger.info(f"  Batch ID: {batch_id}")
        logger.info(f"  Results: {output_file}")
        logger.info(f"{'='*60}\n")

        return output_file

    finally:
        evaluator.close()


def run_forward_evaluation(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = None,
    evaluation_excel: str = None,
    output_dir: str = "./evaluation_output",
    min_citation_count: int = 10,
    seed_paper_title: str = "Attention is All you Need",
    mode: str = "stats",
) -> Dict[str, Any]:
    """
    Convenience function to run complete forward-only evaluation.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        evaluation_excel: Path to manual evaluation Excel file (optional)
        output_dir: Directory to save output files
        min_citation_count: Only consider papers with citation_count > this value (default: 10)
        seed_paper_title: Title of the seed paper (always included regardless of citation count)
        mode: Evaluation mode - "stats" or "metrics" (affects output filename)

    Returns:
        Complete evaluation report dictionary
    """
    evaluator = ForwardOnlyEvaluator(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        seed_paper_title=seed_paper_title,
    )

    try:
        # Load evaluation data if provided
        evaluation_df = None
        if evaluation_excel and Path(evaluation_excel).exists():
            evaluation_df = pd.read_excel(evaluation_excel)
            logger.info(f"Loaded {len(evaluation_df)} samples from {evaluation_excel}")

        # Generate report (with citation count filter)
        report = evaluator.generate_full_report(
            evaluation_df=evaluation_df,
            min_citation_count=min_citation_count,
        )

        # Print report
        text_report = evaluator.print_report(report)
        print(text_report)

        # Export if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Use mode-specific filename prefix
            file_prefix = f"forward_{mode}_report"

            # Export JSON
            json_path = output_path / f"{file_prefix}_{timestamp}.json"
            evaluator.export_to_json(report, str(json_path))

            # Save text report
            report_path = output_path / f"{file_prefix}_{timestamp}.txt"
            with open(report_path, "w") as f:
                f.write(text_report)
            logger.info(f"Text report saved to {report_path}")

            # Generate plots for metrics mode (when manual evaluation data is available)
            if mode == "metrics" and report.get("manual_evaluation"):
                plot_files = evaluator.plot_evaluation_metrics(report, output_dir=output_dir)
                logger.info(f"Generated {len(plot_files)} evaluation plots")

        return report

    finally:
        evaluator.close()


def main():
    """Main entry point for command-line usage."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Forward-Only Evaluation for Semantic Relationship Extraction"
    )
    parser.add_argument(
        "--mode",
        choices=["stats", "automated", "metrics"],
        default="stats",
        help="Evaluation mode: stats (dataset statistics), automated (Claude evaluation), metrics (calculate from Excel)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Sample size for automated evaluation",
    )
    parser.add_argument(
        "--evaluation-excel",
        type=str,
        default=None,
        help="Path to evaluation Excel file for metrics calculation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j connection URI",
    )
    parser.add_argument(
        "--min-citation-count",
        type=int,
        default=10,
        help="Only consider papers with citation_count > this value (default: 10)",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Batch ID to retrieve results from a previous batch (skips sampling and submission)",
    )

    args = parser.parse_args()

    neo4j_password = os.getenv("NEO4J_PASSWORD")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if args.mode == "stats":
        # Run dataset statistics and temporal evolution
        report = run_forward_evaluation(
            neo4j_uri=args.neo4j_uri,
            neo4j_user="neo4j",
            neo4j_password=neo4j_password,
            evaluation_excel=args.evaluation_excel,
            output_dir=args.output_dir,
            min_citation_count=args.min_citation_count,
            mode="stats",
        )
        return report

    elif args.mode == "automated":
        # Run automated evaluation with Claude
        output_file = run_forward_automated_evaluation(
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            neo4j_uri=args.neo4j_uri,
            neo4j_user="neo4j",
            neo4j_password=neo4j_password,
            anthropic_api_key=anthropic_api_key,
            batch_id=args.batch_id,
        )
        return output_file

    elif args.mode == "metrics":
        # Calculate metrics from existing Excel file
        if not args.evaluation_excel:
            print("Error: --evaluation-excel required for metrics mode")
            return None
        report = run_forward_evaluation(
            neo4j_uri=args.neo4j_uri,
            neo4j_user="neo4j",
            neo4j_password=neo4j_password,
            evaluation_excel=args.evaluation_excel,
            output_dir=args.output_dir,
            min_citation_count=args.min_citation_count,
            mode="metrics",
        )
        return report


if __name__ == "__main__":
    main()
