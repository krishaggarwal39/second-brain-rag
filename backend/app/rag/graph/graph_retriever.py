import re
import asyncio
from typing import List, Optional, Set

from app.rag.graph.neo4j_manager import get_neo4j_manager
from app.core.logging import get_logger

logger = get_logger(__name__)

# Minimum candidate entity name length to consider during question matching (R-48)
MIN_ENTITY_NAME_LENGTH: int = 4


def _normalize_for_matching(text: str) -> str:
    """Normalize text for robust deterministic entity matching."""
    text = text.replace("\u2011", "-")
    # Strip possessive 's or ’s (e.g. Darcy's -> Darcy)
    text = re.sub(r"['\u2019]s?\b", "", text)
    # Convert hyphens to spaces to align hyphenated terms with spaced node names
    text = text.replace("-", " ")
    # Collapse whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def find_matching_entities(
    question: str,
    entity_names: List[str],
    max_entities: int = 3,
    min_length: int = MIN_ENTITY_NAME_LENGTH,
) -> List[str]:
    """
    Match candidate entity names against the question locally and deterministically.
    - Case-insensitive
    - Whole-word boundary substring matching (with optional plural suffix)
    - Ignores candidate names shorter than min_length (default MIN_ENTITY_NAME_LENGTH = 4)
    - Prefers longer (more specific) names, returning up to max_entities (default 3)
    """
    q_norm = _normalize_for_matching(question)
    if not q_norm:
        return []

    matched: List[str] = []
    for name in entity_names:
        if len(name.strip()) < min_length:
            continue

        n_norm = _normalize_for_matching(name)
        if not n_norm:
            continue

        # Use lookaround boundaries to match whole words/phrases; s? handles plurals
        pattern = r"(?<!\w)" + re.escape(n_norm) + r"s?(?!\w)"
        if re.search(pattern, q_norm):
            matched.append(name)

    # Sort longest name first (more specific matches prioritized)
    matched.sort(key=lambda x: len(x), reverse=True)
    return matched[:max_entities]


async def retrieve_graph_context(question: str, owner_id: Optional[str] = None) -> str:
    """
    Fast and robust deterministic GraphRAG retriever:
    1. Loads cached entity names for owner_id from Neo4j (no per-query LLM call).
    2. Deterministically matches question against node names locally.
    3. Fetches 1-hop relationship context from Neo4j for top matched entities of owner_id.
    4. Formats and deduplicates context blocks; returns empty string if no match.
    Fully non-fatal: any exception logs and safely returns "".
    """
    if not question or not owner_id:
        return ""

    try:
        manager = get_neo4j_manager()
        entity_names = await asyncio.to_thread(manager.get_all_entity_names, owner_id=str(owner_id))
        if not entity_names:
            return ""

        matched_entities = find_matching_entities(question, entity_names, max_entities=3)
        if not matched_entities:
            return ""

        logger.info(f"Graph Search matched entities for owner {owner_id}: {matched_entities}")

        blocks: List[str] = []
        seen_rel_lines: Set[str] = set()

        for entity_name in matched_entities:
            relationships = await asyncio.to_thread(
                manager.get_related_context, entity_name, str(owner_id)
            )
            if not relationships:
                continue

            lines = [f"KNOWLEDGE GRAPH CONTEXT FOR '{entity_name}':"]
            has_new = False
            for rel in relationships:
                source = rel.get("source", "")
                rel_type = rel.get("relationship", "")
                target = rel.get("target", "")
                ctx = rel.get("context", "")
                line = f"- {source} [{rel_type}] {target} (Context: {ctx})"
                if line not in seen_rel_lines:
                    seen_rel_lines.add(line)
                    lines.append(line)
                    has_new = True

            if has_new:
                blocks.append("\n".join(lines))

        if not blocks:
            return ""

        return "\n\n".join(blocks)

    except Exception as e:
        logger.error(f"Graph retrieval failed: {e}")
        return ""
