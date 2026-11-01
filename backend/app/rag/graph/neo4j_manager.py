import os
import re
import time
from typing import List, Optional
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
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info(f"Initialized Neo4j driver for {uri} (lazy connection).")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")

        self._entity_names_cache: Optional[List[str]] = None
        self._entity_names_cache_time: float = 0.0
        self._cache_ttl: float = float(os.getenv("GRAPH_ENTITY_CACHE_TTL", "300"))

    def close(self):
        if self.driver:
            self.driver.close()
        self._entity_names_cache = None

    @_neo4j_retry
    def add_entity(self, label: str, name: str, description: str = ""):
        if not self.driver or not name:
            return

        clean_label = re.sub(r"[^a-zA-Z0-9_]", "", label) or "Entity"

        query = (
            f"MERGE (e:{clean_label} {{name: $name}}) "
            "ON CREATE SET e.description = $description "
            "RETURN e"
        )
        with self.driver.session() as session:
            session.run(query, name=name, description=description)
        self._entity_names_cache = None

    @_neo4j_retry
    def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relationship_type: str,
        context: str = "",
    ):
        if not self.driver or not source_name or not target_name:
            return

        clean_type = (
            re.sub(r"[^a-zA-Z0-9_]", "", relationship_type.upper().replace(" ", "_"))
            or "RELATED_TO"
        )

        query = (
            "MATCH (a {name: $source}) "
            "MATCH (b {name: $target}) "
            f"MERGE (a)-[r:{clean_type}]->(b) "
            "ON CREATE SET r.context = $context "
            "RETURN type(r)"
        )
        with self.driver.session() as session:
            session.run(query, source=source_name, target=target_name, context=context)

    @_neo4j_retry
    def get_related_context(self, entity_name: str) -> list:
        """Returns 1-hop relationships for an entity to build graph context."""
        if not self.driver:
            return []

        query = (
            "MATCH (a {name: $name})-[r]-(b) "
            "RETURN a.name AS source, type(r) AS relationship, b.name AS target, r.context AS context "
            "LIMIT 10"
        )
        results = []
        with self.driver.session() as session:
            for record in session.run(query, name=entity_name):
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
    def _fetch_all_entity_names_cypher(self) -> List[str]:
        if not self.driver:
            return []
        query = "MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name"
        names = []
        with self.driver.session() as session:
            for record in session.run(query):
                val = record.get("name")
                if val:
                    names.append(str(val))
        return names

    def get_all_entity_names(self, refresh: bool = False) -> List[str]:
        """Fetch all distinct node names from Neo4j with instance-level TTL caching.
        Resilient: returns [] on error so graph remains non-fatal.
        """
        now = time.time()
        if (
            not refresh
            and self._entity_names_cache is not None
            and (now - self._entity_names_cache_time) < self._cache_ttl
        ):
            return list(self._entity_names_cache)

        if not self.driver:
            return []

        try:
            names = self._fetch_all_entity_names_cypher()
            self._entity_names_cache = names
            self._entity_names_cache_time = now
            return list(names)
        except Exception as e:
            logger.error(f"Failed to fetch entity names from Neo4j: {e}")
            return list(self._entity_names_cache) if self._entity_names_cache is not None else []


# Singleton
_manager: Neo4jManager = None


def get_neo4j_manager() -> Neo4jManager:
    global _manager
    if _manager is None:
        _manager = Neo4jManager()
    return _manager
