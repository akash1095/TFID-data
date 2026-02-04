from typing import List, Literal

from pydantic import BaseModel, Field

# Valid relationship types for AI/ML citation networks
RELATIONSHIP_TYPES = [
    "Implements",  # Uses cited method as-is
    "Evaluates-On",  # Uses cited dataset/benchmark
    "Builds-On",  # Uses as base, adds components
    "Extends",  # Modifies internals (same domain)
    "Adapts",  # Transfers to different domain
    "Outperforms",  # Shows superior results
    "Compares-With",  # Neutral baseline comparison
    "Contradicts",  # Contradicts with evidence
    "Analyzes",  # Studies/investigates
    "Surveys",  # Reviews in literature survey
]


class Relationship(BaseModel):
    """Represents a single semantic relationship between two research papers."""

    type: str = Field(
        ...,
        description="The type of semantic relationship (e.g., Implements, Evaluates-On, Builds-On, Extends, Adapts, Outperforms, Compares-With, Contradicts, Analyzes, Surveys)",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence level of the identified relationship"
    )
    evidence: str = Field(
        ...,
        description="Specific text or reasoning from the abstracts supporting this relationship in 20 words",
    )
    explanation: str = Field(
        ...,
        description="Brief explanation of why this relationship exists between the papers in 20 words",
    )


class RelationshipAnalysis(BaseModel):
    """Complete analysis of semantic relationships between two research papers."""

    relationships: List[Relationship] = Field(
        default_factory=list,
        description="List of identified semantic relationships between the papers",
    )
    # no_relationship_reason: Optional[str] = Field(
    #     default=None,
    #     description="Explanation for why no relationships were found, if applicable",
    # )
