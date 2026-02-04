EXTRACT_PROMPT_NEW = """You are analyzing research papers. Find relationships between NEW and OLD paper.

**NEW PAPER (Citing):**
Title: {citing_title}
Abstract: {citing_abstract}

**OLD PAPER (Cited):**
Title: {cited_title}
Abstract: {cited_abstract}

---

**STEP 0: CHECK FOR NO RELATIONSHIP FIRST**

Output NO RELATIONSHIP if ANY of these are true:
- Papers work on DIFFERENT topics/domains with no overlap
- Papers work on SAME topic but NEW doesn't mention OLD's specific method/dataset/finding
- NEW paper's abstract has NO explicit reference to OLD paper's contribution
- Connection is only thematic (both do "NLP" or both do "vision" is NOT enough)
- You cannot find a direct quote in NEW that references OLD

Examples of NO RELATIONSHIP:
- NEW: image classification, OLD: knowledge graphs → No Relationship
- NEW: "We propose a new attention mechanism", OLD: "We propose BERT" (no mention of BERT) → No Relationship
- NEW and OLD both do NLP but NEW doesn't use OLD's method → No Relationship

If NO RELATIONSHIP → Output:
{{"relationships": [{{"type": "No-Relation", "confidence": "high", "evidence": "No explicit connection found", "explanation": "Specific reason why no connection exists"}}]}}

---

**STEP 1: If relationship exists, check Extends or Analyzes FIRST**

**Extends** = NEW IMPROVES/MODIFIES OLD's method (VERY COMMON in ML papers)
- Keywords: "improve", "modify", "enhance", "extend", "generalize", "variant", "more efficient", "better"
- NEW changes OLD's internals: architecture, loss function, training, algorithm
- Same problem domain as OLD
- Examples:
  - "We improve the attention mechanism of Transformers" → Extends
  - "We propose a more efficient variant of BERT" → Extends

**Analyzes** = NEW STUDIES/INVESTIGATES OLD (no new method)
- Keywords: "analyze", "study", "investigate", "probe", "understand", "examine", "interpret"
- NEW's goal is to UNDERSTAND OLD, not build something new
- Examples:
  - "We analyze why BERT works so well" → Analyzes
  - "We study the attention patterns in Transformers" → Analyzes

---

**STEP 2: Only if NOT Extends or Analyzes, check these:**

**Builds-On** = NEW uses OLD as-is, adds SEPARATE external component
- OLD is used as a BLACK BOX (unchanged)
- NEW adds something EXTERNAL on top
- LESS COMMON than Extends
- Example: "We use pre-trained BERT and add a classification head" → Builds-On
- CAREFUL: If NEW modifies OLD's internals → use Extends instead!

**Outperforms** = NEW beats OLD with EXPLICIT NUMBERS
- REQUIRES: Numerical comparison (e.g., "95% vs 87%")
- REQUIRES: Clear statement NEW is better

**Other types:**
- Implements: uses method as-is (no modification, no addition)
- Evaluates-On: uses dataset/benchmark
- Adapts: transfers to different domain (NLP→Vision)
- Compares-With: neutral comparison (no winner)
- Contradicts: contradicts with evidence
- Surveys: reviews in literature survey

---

**DECISION GUIDE:**

Is there ANY explicit connection to OLD's contribution?
├─ NO → **No Relationship** (STOP HERE)
└─ YES → continue

Does NEW modify/improve HOW OLD works?
├─ YES → **Extends**
└─ NO → continue

Does NEW study/analyze OLD without building new method?
├─ YES → **Analyzes**
└─ NO → continue

Does NEW use OLD unchanged and add external component?
├─ YES → **Builds-On**
└─ NO → check other types

---

**OUTPUT (JSON ONLY):**

{{
  "relationships": [
    {{
      "type": "Extends",
      "confidence": "high",
      "evidence": "Quote from abstract",
      "explanation": "Why this relationship"
    }}
  ]
}}

If no relationship:
{{
  "relationships": [
    {{
      "type": "No-Relation",
      "confidence": "high",
      "evidence": "No explicit connection found",
      "explanation": "Papers address different topics with no direct connection"
    }}
  ]
}}
"""


# =============================================================================
# LLAMA 3.1 8B OPTIMIZED PROMPTS
# =============================================================================

LLAMA_8B_SYSTEM_PROMPT = """You are an research paper abstract analyst. You extract semantic citation relationships between papers.
You ALWAYS respond with valid JSON only. No explanations, no markdown, just JSON."""


LLAMA_8B_EXTRACT_PROMPT = """Analyze how the CITING paper relates to the CITED paper.

CITING PAPER:
Title: {citing_title}
Abstract: {citing_abstract}

CITED PAPER:
Title: {cited_title}
Abstract: {cited_abstract}



RELATIONSHIP TYPES (pick all that apply):
* Implements: Uses the cited method or algorithm directly without modification
* Builds-On: Uses the cited work as a base and adds new components or modules on top without altering its core
* Extends: Modifies, Build upon or improves the methodology, framework, or theoretical approach from the cited paper
* Adapts: Applies the same core method or concept to a different domain, task, or modality
* Outperforms: The citing paper directly compares its method against the cited paper's method on the same task and reports superior quantitative performance
* Compares-With: Uses the cited method as a baseline for neutral or mixed-performance comparison
* Contradicts: Presents evidence contradicting cited paper's claims
* Analyzes: Investigates or analyzes the properties, behavior, limitations, or characteristics of the cited method
* Surveys: Reviews or categorizes the cited work in a literature survey
* Evaluates-On: Evaluates its method on a dataset or benchmark introduced in the cited work

If None of the above relationships apply, respond with No-Relation.
No-Relation: If the papers are on different topics or there is no explicit reference to the cited paper's contribution

KEY DISTINCTIONS:
- Implements vs Builds-On: Implements uses as-is; Builds-On adds components
- Builds-On vs Extends: Builds-On adds externally; Extends modifies internals
- Extends vs Adapts: Extends same domain; Adapts crosses domains
- Outperforms vs Compares-With: Outperforms claims superiority; Compares-With is neutral

RULES:
1. Only identify relationships with clear evidence between CITING and CITED papers
2. Quote specific text as evidence and explanation in 20 words each


Respond with this exact JSON format:
{{"relationships": [{{"type": "TypeName", "confidence": "high|medium|low", "evidence": "quote from abstract", "explanation": "why this relationship exists"}}]}}"""
