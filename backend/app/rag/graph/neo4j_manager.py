import os
import re
import time
from typing import Dict, List, Optional, Tuple
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.logging import get_logger

logger = get_logger(__name__)

# Retry on transient Neo4j errors (connection drops, leader re-election, etc.)
_neo4j_retry = retry(
    stop=stop_after_attempt(int(os.getenv("DB_MAX_RETRIES", "3"))),
    wait=wait_exponential(multiplier=1, min=0.5, max=5.0),
    retry=retry_if_exception_type((ServiceUnavailable, SessionExpired, TransientError, OSError)),
    reraise=True,
)


class Neo4jManager:
    def __init__(self, driver=None):
        if driver is not None:
            self.driver = driver
        else:
            uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")

            self.driver = None
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                logger.info(f"Initialized Neo4j driver for {uri} (lazy connection).")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")

        # Per-owner entity names cache: owner_id -> (names_list, timestamp)
        self._entity_names_cache: Dict[str, Tuple[List[str], float]] = {}
        self._cache_ttl: float = float(os.getenv("GRAPH_ENTITY_CACHE_TTL", "300"))

    def close(self):
        if self.driver:
            self.driver.close()
        self._entity_names_cache.clear()

    def invalidate_entity_cache(self, owner_id: Optional[str] = None):
        if owner_id:
            self._entity_names_cache.pop(str(owner_id), None)
        else:
            self._entity_names_cache.clear()

    @_neo4j_retry
    def add_entity(
        self,
        label: str,
        name: str,
        description: str = "",
        owner_id: str = "",
        doc_id: Optional[str] = None,
    ):
        if not self.driver or not name or not owner_id:
            return

        clean_label = re.sub(r"[^a-zA-Z0-9_]", "", label) or "Entity"

        query = (
            f"MERGE (e:{clean_label} {{name: $name, owner_id: $owner_id}}) "
            "ON CREATE SET "
            "e.description = $description, "
            "e.doc_id = $doc_id, "
            "e.doc_ids = CASE WHEN $doc_id IS NOT NULL AND $doc_id <> '' THEN [$doc_id] ELSE [] END "
            "ON MATCH SET "
            "e.doc_id = CASE WHEN $doc_id IS NOT NULL AND $doc_id <> '' THEN $doc_id ELSE e.doc_id END, "
            "e.doc_ids = CASE "
            "WHEN $doc_id IS NOT NULL AND $doc_id <> '' AND NOT $doc_id IN coalesce(e.doc_ids, []) "
            "THEN coalesce(e.doc_ids, []) + [$doc_id] "
            "ELSE coalesce(e.doc_ids, []) END, "
            "e.description = CASE WHEN (e.description IS NULL OR e.description = '') AND $description <> '' THEN $description ELSE e.description END "
            "RETURN e"
        )
        with self.driver.session() as session:
            session.run(
                query,
                name=name,
                description=description,
                owner_id=str(owner_id),
                doc_id=str(doc_id) if doc_id else "",
            )
        self.invalidate_entity_cache(str(owner_id))

    @_neo4j_retry
    def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relationship_type: str,
        context: str = "",
        owner_id: str = "",
        doc_id: Optional[str] = None,
    ):
        if not self.driver or not source_name or not target_name or not owner_id:
            return

        clean_type = (
            re.sub(r"[^a-zA-Z0-9_]", "", relationship_type.upper().replace(" ", "_"))
            or "RELATED_TO"
        )

        query = (
            "MATCH (a {name: $source, owner_id: $owner_id}) "
            "MATCH (b {name: $target, owner_id: $owner_id}) "
            f"MERGE (a)-[r:{clean_type} {{owner_id: $owner_id}}]->(b) "
            "ON CREATE SET "
            "r.context = $context, "
            "r.doc_id = $doc_id, "
            "r.doc_ids = CASE WHEN $doc_id IS NOT NULL AND $doc_id <> '' THEN [$doc_id] ELSE [] END "
            "ON MATCH SET "
            "r.doc_id = CASE WHEN $doc_id IS NOT NULL AND $doc_id <> '' THEN $doc_id ELSE r.doc_id END, "
            "r.doc_ids = CASE "
            "WHEN $doc_id IS NOT NULL AND $doc_id <> '' AND NOT $doc_id IN coalesce(r.doc_ids, []) "
            "THEN coalesce(r.doc_ids, []) + [$doc_id] "
            "ELSE coalesce(r.doc_ids, []) END, "
            "r.context = CASE WHEN (r.context IS NULL OR r.context = '') AND $context <> '' THEN $context ELSE r.context END "
            "RETURN type(r)"
        )
        with self.driver.session() as session:
            session.run(
                query,
                source=source_name,
                target=target_name,
                context=context,
                owner_id=str(owner_id),
                doc_id=str(doc_id) if doc_id else "",
            )

    @_neo4j_retry
    def get_related_context(self, entity_name: str, owner_id: str) -> list:
        """Returns 1-hop relationships for an entity to build graph context for a specific owner."""
        if not self.driver or not entity_name or not owner_id:
            return []

        query = (
            "MATCH (a {name: $name, owner_id: $owner_id})-[r {owner_id: $owner_id}]-(b {owner_id: $owner_id}) "
            "RETURN a.name AS source, type(r) AS relationship, b.name AS target, r.context AS context "
            "LIMIT 10"
        )
        results = []
        with self.driver.session() as session:
            for record in session.run(query, name=entity_name, owner_id=str(owner_id)):
                results.append(
                    {
                        "source": record["source"],
                        "relationship": record["relationship"],
                        "target": record["target"],
                        "context": record.get("context", ""),
                    }
                )
        return results

    @_neo4j_retry
    def _fetch_all_entity_names_cypher(self, owner_id: str) -> List[str]:
        if not self.driver or not owner_id:
            return []
        query = (
            "MATCH (n {owner_id: $owner_id}) "
            "WHERE n.name IS NOT NULL "
            "RETURN DISTINCT n.name AS name"
        )
        names = []
        with self.driver.session() as session:
            for record in session.run(query, owner_id=str(owner_id)):
                val = record.get("name")
                if val:
                    names.append(str(val))
        return names

    def get_all_entity_names(self, owner_id: str, refresh: bool = False) -> List[str]:
        """Fetch all distinct node names for a specific owner from Neo4j with per-owner TTL caching.
        Resilient: returns [] on error so graph remains non-fatal.
        """
        if not owner_id:
            return []

        owner_key = str(owner_id)
        now = time.time()
        cached = self._entity_names_cache.get(owner_key)
        if not refresh and cached is not None:
            cached_names, cache_time = cached
            if (now - cache_time) < self._cache_ttl:
                return list(cached_names)

        if not self.driver:
            return []

        try:
            names = self._fetch_all_entity_names_cypher(owner_id=owner_key)
            self._entity_names_cache[owner_key] = (names, now)
            return list(names)
        except Exception as e:
            logger.error(f"Failed to fetch entity names from Neo4j for owner {owner_key}: {e}")
            if owner_key in self._entity_names_cache:
                return list(self._entity_names_cache[owner_key][0])
            return []

    @_neo4j_retry
    def delete_document_graph(self, doc_id: str, owner_id: str):
        """Remove doc_id from relationships and nodes for owner_id.
        Delete relationships and orphan nodes when their doc_ids list becomes empty.
        """
        if not self.driver or not doc_id or not owner_id:
            return

        with self.driver.session() as session:
            # 1. Update relationships: remove doc_id, delete if doc_ids becomes empty
            session.run(
                """
                MATCH ()-[r {owner_id: $owner_id}]->()
                WHERE $doc_id IN coalesce(r.doc_ids, []) OR r.doc_id = $doc_id
                WITH r, [d IN coalesce(r.doc_ids, []) WHERE d <> $doc_id] AS remaining_docs
                SET r.doc_ids = remaining_docs,
                    r.doc_id = CASE WHEN size(remaining_docs) > 0 THEN remaining_docs[0] ELSE NULL END
                WITH r, remaining_docs
                WHERE size(remaining_docs) = 0
                DELETE r
                """,
                owner_id=str(owner_id),
                doc_id=str(doc_id),
            )

            # 2. Update nodes: remove doc_id
            session.run(
                """
                MATCH (n {owner_id: $owner_id})
                WHERE $doc_id IN coalesce(n.doc_ids, []) OR n.doc_id = $doc_id
                WITH n, [d IN coalesce(n.doc_ids, []) WHERE d <> $doc_id] AS remaining_docs
                SET n.doc_ids = remaining_docs,
                    n.doc_id = CASE WHEN size(remaining_docs) > 0 THEN remaining_docs[0] ELSE NULL END
                """,
                owner_id=str(owner_id),
                doc_id=str(doc_id),
            )

            # 3. Delete orphan nodes whose doc_ids is empty
            session.run(
                """
                MATCH (n {owner_id: $owner_id})
                WHERE NOT (n)--() AND size(coalesce(n.doc_ids, [])) = 0
                DELETE n
                """,
                owner_id=str(owner_id),
            )

        self.invalidate_entity_cache(str(owner_id))

    delete_document = delete_document_graph

    @_neo4j_retry
    def purge_legacy_graph(self) -> int:
        """Maintenance helper: deletes all nodes (and their connected relationships) that have no owner_id.
        Idempotent. Does NOT run on startup.
        Returns the number of legacy nodes deleted.
        """
        if not self.driver:
            return 0

        query = (
            "MATCH (n) "
            "WHERE n.owner_id IS NULL OR n.owner_id = '' "
            "DETACH DELETE n "
            "RETURN count(n) AS deleted_count"
        )
        deleted = 0
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                deleted = record.get("deleted_count", 0)

        self._entity_names_cache.clear()
        logger.info(f"Purged {deleted} legacy graph nodes without owner_id")
        return deleted


# Singleton
_manager: Optional[Neo4jManager] = None


def get_neo4j_manager() -> Neo4jManager:
    global _manager
    if _manager is None:
        _manager = Neo4jManager()
    return _manager


def reset_neo4j_manager():
    global _manager
    if _manager is not None:
        _manager.close()
    _manager = None


def purge_legacy_graph() -> int:
    """Maintenance helper to purge legacy graph nodes without owner_id."""
    return get_neo4j_manager().purge_legacy_graph()
