import json
import asyncio
from typing import Optional
from app.rag.graph.neo4j_manager import get_neo4j_manager
from app.core.logging import get_logger
from app.core.llm_manager import llm_manager

logger = get_logger(__name__)

EXTRACTION_PROMPT = """\
You are a highly intelligent Knowledge Graph extractor.
Given the following text, extract the main Entities (nodes) and their Relationships (edges).
Return ONLY a valid JSON object with the following structure:
{
    "entities": [
        {"name": "Entity Name", "type": "Person/Organization/Concept", "description": "Brief context"}
    ],
    "relationships": [
        {"source": "Entity 1", "target": "Entity 2", "type": "WORKS_FOR, USES, RELATED_TO, etc", "context": "Why they are related"}
    ]
}
If no relevant entities are found, return empty lists.
"""


async def extract_and_store_graph(
    text: str,
    owner_id: Optional[str] = None,
    doc_id: Optional[str] = None,
):
    """Extracts a knowledge graph from text using the unified LLM interface and stores it in Neo4j."""
    if not owner_id:
        logger.warning("Graph extraction skipped: no owner_id provided")
        return

    result_text = await llm_manager.generate(
        system_prompt=EXTRACTION_PROMPT,
        user_content=f"Text:\n{text}",
        json_mode=True,
        timeout=30.0,
    )

    if not result_text:
        return

    try:
        graph_data = json.loads(result_text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Graph extraction: failed to parse JSON: {e}")
        return

    if not isinstance(graph_data, dict):
        logger.warning("Graph extraction: root JSON is not an object")
        return

    manager = get_neo4j_manager()

    raw_entities = graph_data.get("entities")
    valid_entities = []
    if isinstance(raw_entities, list):
        for entity in raw_entities:
            if not isinstance(entity, dict):
                continue
            name = entity.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            label = entity.get("type", "Entity")
            if not isinstance(label, str):
                label = "Entity"
            label = label.replace(" ", "")
            description = entity.get("description", "")
            if not isinstance(description, str):
                description = ""
            valid_entities.append((label, name, description))

    for label, name, description in valid_entities:
        await asyncio.to_thread(
            manager.add_entity,
            label=label,
            name=name,
            description=description,
            owner_id=str(owner_id),
            doc_id=doc_id,
        )

    raw_relationships = graph_data.get("relationships")
    valid_relationships = []
    if isinstance(raw_relationships, list):
        for rel in raw_relationships:
            if not isinstance(rel, dict):
                continue
            source = rel.get("source")
            target = rel.get("target")
            if not isinstance(source, str) or not source.strip():
                continue
            if not isinstance(target, str) or not target.strip():
                continue
            source = source.strip()
            target = target.strip()
            rel_type = rel.get("type", "RELATED_TO")
            if not isinstance(rel_type, str):
                rel_type = "RELATED_TO"
            rel_type = (
                rel_type.upper()
                .replace(" ", "_")
                .replace("-", "_")
            )
            context = rel.get("context", "")
            if not isinstance(context, str):
                context = ""
            valid_relationships.append((source, target, rel_type, context))

    for source, target, rel_type, context in valid_relationships:
        await asyncio.to_thread(
            manager.add_relationship,
            source_name=source,
            target_name=target,
            relationship_type=rel_type,
            context=context,
            owner_id=str(owner_id),
            doc_id=doc_id,
        )

    logger.info(
        f"Graph extraction stored {len(valid_entities)} entities "
        f"and {len(valid_relationships)} relationships for owner {owner_id}."
    )
