import os
import pytest
from unittest.mock import AsyncMock, patch

from app.rag.graph.neo4j_manager import Neo4jManager, get_neo4j_manager, purge_legacy_graph
from app.rag.graph.graph_retriever import (
    retrieve_graph_context,
    find_matching_entities,
    MIN_ENTITY_NAME_LENGTH,
)
from app.rag.graph.graph_extractor import extract_and_store_graph


class FakeSession:
    """In-memory fake Neo4j session that records Cypher + params and simulates graph state."""

    def __init__(self, storage):
        self.storage = storage

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(self, query: str, **params):
        self.storage["recorded_calls"].append((query, params))
        normalized = " ".join(query.strip().split())

        # 1. MERGE entity
        if normalized.startswith("MERGE (e:") and "owner_id: $owner_id" in query:
            name = params.get("name")
            owner_id = str(params.get("owner_id", ""))
            desc = params.get("description", "")
            doc_id = params.get("doc_id", "")
            key = (name, owner_id)
            if key not in self.storage["nodes"]:
                self.storage["nodes"][key] = {
                    "name": name,
                    "owner_id": owner_id,
                    "description": desc,
                    "doc_id": doc_id,
                    "doc_ids": [doc_id] if doc_id else [],
                }
            else:
                node = self.storage["nodes"][key]
                if doc_id and doc_id not in node["doc_ids"]:
                    node["doc_ids"].append(doc_id)
                if doc_id:
                    node["doc_id"] = doc_id
            return []

        # 2. MATCH a, MATCH b, MERGE relationship
        if "MERGE (a)-[r:" in query and "owner_id: $owner_id" in query:
            source = params.get("source")
            target = params.get("target")
            owner_id = str(params.get("owner_id", ""))
            context = params.get("context", "")
            doc_id = params.get("doc_id", "")

            import re
            m = re.search(r"MERGE \(a\)-\[r:([A-Za-z0-9_]+)", query)
            rel_type = m.group(1) if m else "RELATED_TO"

            # Ensure both source and target exist for this owner
            if (source, owner_id) in self.storage["nodes"] and (target, owner_id) in self.storage["nodes"]:
                rel_key = (source, target, owner_id)
                if rel_key not in self.storage["relationships"]:
                    self.storage["relationships"][rel_key] = {
                        "source": source,
                        "target": target,
                        "relationship": rel_type,
                        "context": context,
                        "owner_id": owner_id,
                        "doc_id": doc_id,
                        "doc_ids": [doc_id] if doc_id else [],
                    }
                else:
                    rel = self.storage["relationships"][rel_key]
                    if doc_id and doc_id not in rel["doc_ids"]:
                        rel["doc_ids"].append(doc_id)
                    if doc_id:
                        rel["doc_id"] = doc_id
            return []

        # 3. get_related_context: MATCH (a {name: $name, owner_id: $owner_id})-[r {owner_id: $owner_id}]-(b {owner_id: $owner_id})
        if "get_related_context" in query or ("-[r {owner_id: $owner_id}]-" in query and "LIMIT 10" in query):
            name = params.get("name")
            owner_id = str(params.get("owner_id", ""))
            results = []
            for (src, tgt, r_owner), rel in self.storage["relationships"].items():
                if r_owner == owner_id and (src == name or tgt == name):
                    results.append(
                        {
                            "source": rel["source"],
                            "relationship": rel.get("relationship", "RELATED_TO"),
                            "target": rel["target"],
                            "context": rel.get("context", ""),
                        }
                    )
            return results

        # 4. _fetch_all_entity_names_cypher: MATCH (n {owner_id: $owner_id})
        if "MATCH (n {owner_id: $owner_id})" in query and "RETURN DISTINCT n.name" in query:
            owner_id = str(params.get("owner_id", ""))
            names = []
            for (name, n_owner) in self.storage["nodes"].keys():
                if n_owner == owner_id:
                    names.append({"name": name})
            return names

        # 5. Delete relationships for doc_id and owner_id
        if "MATCH ()-[r {owner_id: $owner_id}]->()" in query and "DELETE r" in query:
            owner_id = str(params.get("owner_id", ""))
            doc_id = str(params.get("doc_id", ""))
            to_delete = []
            for key, rel in list(self.storage["relationships"].items()):
                if rel["owner_id"] == owner_id:
                    if doc_id in rel.get("doc_ids", []):
                        rel["doc_ids"] = [d for d in rel["doc_ids"] if d != doc_id]
                    elif rel.get("doc_id") == doc_id:
                        rel["doc_ids"] = []
                    if not rel["doc_ids"]:
                        to_delete.append(key)
            for key in to_delete:
                del self.storage["relationships"][key]
            return []

        # 6. Delete/update nodes for doc_id and owner_id
        if "MATCH (n {owner_id: $owner_id})" in query and "remaining_docs" in query:
            owner_id = str(params.get("owner_id", ""))
            doc_id = str(params.get("doc_id", ""))
            for (name, n_owner), node in list(self.storage["nodes"].items()):
                if n_owner == owner_id:
                    if doc_id in node.get("doc_ids", []):
                        node["doc_ids"] = [d for d in node["doc_ids"] if d != doc_id]
                    elif node.get("doc_id") == doc_id:
                        node["doc_ids"] = []
            return []

        # 7. Delete orphan nodes
        if "MATCH (n {owner_id: $owner_id})" in query and "WHERE NOT (n)--()" in query:
            owner_id = str(params.get("owner_id", ""))
            to_delete = []
            for (name, n_owner), node in list(self.storage["nodes"].items()):
                if n_owner == owner_id and not node.get("doc_ids"):
                    # Check if node has any relationships
                    has_rel = any(
                        (r["source"] == name or r["target"] == name) and r["owner_id"] == owner_id
                        for r in self.storage["relationships"].values()
                    )
                    if not has_rel:
                        to_delete.append((name, n_owner))
            for key in to_delete:
                del self.storage["nodes"][key]
            return []

        # 8. purge_legacy_graph: MATCH (n) WHERE n.owner_id IS NULL OR n.owner_id = ''
        if "WHERE n.owner_id IS NULL OR n.owner_id = ''" in query and "DETACH DELETE n" in query:
            to_delete_nodes = [k for k, node in self.storage["nodes"].items() if not node.get("owner_id")]
            count = len(to_delete_nodes)
            for k in to_delete_nodes:
                del self.storage["nodes"][k]
            to_delete_rels = [k for k, rel in self.storage["relationships"].items() if not rel.get("owner_id")]
            for k in to_delete_rels:
                del self.storage["relationships"][k]

            class SingleRecord:
                def single(self):
                    return {"deleted_count": count}

            return SingleRecord()

        return []


class FakeNeo4jDriver:
    """Fake Neo4j driver that supplies FakeSession."""

    def __init__(self):
        self.storage = {
            "nodes": {},
            "relationships": {},
            "recorded_calls": [],
        }

    def session(self):
        return FakeSession(self.storage)

    def close(self):
        pass


def test_named_constant_min_entity_name_length():
    """R-48: MIN_ENTITY_NAME_LENGTH must be a named constant equal to 4."""
    assert MIN_ENTITY_NAME_LENGTH == 4
    # Short names (< 4) skipped
    assert find_matching_entities("Who created AI and Python?", ["AI", "Python"]) == ["Python"]
    assert find_matching_entities("Tell me about RAG and Neo4j", ["RAG", "Neo4j"]) == ["Neo4j"]


@pytest.mark.asyncio
async def test_every_query_includes_owner_parameter():
    """All Cypher operations must include owner_id in Cypher query string and query parameters."""
    driver = FakeNeo4jDriver()
    manager = Neo4jManager(driver=driver)

    manager.add_entity(label="Project", name="Falcon", description="Secret", owner_id="user_a", doc_id="doc_1")
    manager.add_entity(label="Company", name="Zenith", description="Target", owner_id="user_a", doc_id="doc_1")
    manager.add_relationship(
        source_name="Falcon",
        target_name="Zenith",
        relationship_type="ACQUIRES",
        context="budget 40M",
        owner_id="user_a",
        doc_id="doc_1",
    )
    _ = manager.get_related_context(entity_name="Falcon", owner_id="user_a")
    _ = manager.get_all_entity_names(owner_id="user_a")
    manager.delete_document_graph(doc_id="doc_1", owner_id="user_a")

    calls = driver.storage["recorded_calls"]
    assert len(calls) >= 6

    for query, params in calls:
        assert "$owner_id" in query, f"Cypher query missing $owner_id: {query}"
        assert "owner_id" in params, f"Params missing owner_id: {params}"
        assert params["owner_id"] == "user_a"


@pytest.mark.asyncio
async def test_two_user_graph_isolation():
    """Two-user test: User A stores confidential data; User B retrieves nothing.
    Deleting User A's doc removes only User A's data without affecting User B.
    """
    driver = FakeNeo4jDriver()
    manager = Neo4jManager(driver=driver)

    # Ingest User A's graph data
    manager.add_entity("Project", "Project Falcon", "User A secret project", owner_id="user_a", doc_id="doc_a1")
    manager.add_entity("Company", "Zenith Ltd", "Target company", owner_id="user_a", doc_id="doc_a1")
    manager.add_relationship(
        source_name="Project Falcon",
        target_name="Zenith Ltd",
        relationship_type="ACQUIRES",
        context="CONFIDENTIAL: budget 40M",
        owner_id="user_a",
        doc_id="doc_a1",
    )

    # Ingest User B's graph data with a SAME-NAMED entity
    manager.add_entity("Project", "Project Falcon", "User B open initiative", owner_id="user_b", doc_id="doc_b1")
    manager.add_entity("Company", "Alpha Corp", "Partner", owner_id="user_b", doc_id="doc_b1")
    manager.add_relationship(
        source_name="Project Falcon",
        target_name="Alpha Corp",
        relationship_type="PARTNERS_WITH",
        context="Public initiative",
        owner_id="user_b",
        doc_id="doc_b1",
    )

    # Verify nodes did NOT merge
    assert ("Project Falcon", "user_a") in driver.storage["nodes"]
    assert ("Project Falcon", "user_b") in driver.storage["nodes"]
    assert driver.storage["nodes"][("Project Falcon", "user_a")]["description"] == "User A secret project"
    assert driver.storage["nodes"][("Project Falcon", "user_b")]["description"] == "User B open initiative"

    # User B queries "Project Falcon" -> must NEVER see Zenith Ltd or 40M budget
    with patch("app.rag.graph.graph_retriever.get_neo4j_manager", return_value=manager):
        ctx_b = await retrieve_graph_context("Tell me about Project Falcon", owner_id="user_b")
        assert "Zenith Ltd" not in ctx_b
        assert "CONFIDENTIAL" not in ctx_b
        assert "Alpha Corp" in ctx_b
        assert "PARTNERS_WITH" in ctx_b

        # User A queries "Project Falcon" -> sees Zenith Ltd
        ctx_a = await retrieve_graph_context("Tell me about Project Falcon", owner_id="user_a")
        assert "Zenith Ltd" in ctx_a
        assert "CONFIDENTIAL: budget 40M" in ctx_a
        assert "Alpha Corp" not in ctx_a

        # User C (no uploads) queries -> retrieves nothing
        ctx_c = await retrieve_graph_context("Tell me about Project Falcon", owner_id="user_c")
        assert ctx_c == ""

    # Delete User A's document doc_a1
    manager.delete_document_graph(doc_id="doc_a1", owner_id="user_a")

    # User A's data is gone
    assert ("Project Falcon", "user_a") not in driver.storage["nodes"]
    assert ("Zenith Ltd", "user_a") not in driver.storage["nodes"]
    assert ("Project Falcon", "Zenith Ltd", "user_a") not in driver.storage["relationships"]

    # User B's data is completely intact
    assert ("Project Falcon", "user_b") in driver.storage["nodes"]
    assert ("Alpha Corp", "user_b") in driver.storage["nodes"]
    assert ("Project Falcon", "Alpha Corp", "user_b") in driver.storage["relationships"]

    with patch("app.rag.graph.graph_retriever.get_neo4j_manager", return_value=manager):
        ctx_b_after = await retrieve_graph_context("Tell me about Project Falcon", owner_id="user_b")
        assert "Alpha Corp" in ctx_b_after


@pytest.mark.asyncio
async def test_relationship_supported_by_multiple_docs_survives_deleting_one():
    """A relationship supported by several docs of the same owner must survive deleting just one."""
    driver = FakeNeo4jDriver()
    manager = Neo4jManager(driver=driver)

    # Ingest from doc 1
    manager.add_entity("Concept", "FastAPI", "Web framework", owner_id="user_1", doc_id="doc_1")
    manager.add_entity("Concept", "Python", "Language", owner_id="user_1", doc_id="doc_1")
    manager.add_relationship(
        source_name="FastAPI",
        target_name="Python",
        relationship_type="BUILT_ON",
        context="Framework built on Python",
        owner_id="user_1",
        doc_id="doc_1",
    )

    # Ingest same relationship from doc 2
    manager.add_entity("Concept", "FastAPI", "Web framework", owner_id="user_1", doc_id="doc_2")
    manager.add_entity("Concept", "Python", "Language", owner_id="user_1", doc_id="doc_2")
    manager.add_relationship(
        source_name="FastAPI",
        target_name="Python",
        relationship_type="BUILT_ON",
        context="Framework built on Python",
        owner_id="user_1",
        doc_id="doc_2",
    )

    rel = driver.storage["relationships"][("FastAPI", "Python", "user_1")]
    assert set(rel["doc_ids"]) == {"doc_1", "doc_2"}

    # Delete doc 1 -> relationship and nodes must SURVIVE because doc 2 still supports them
    manager.delete_document_graph(doc_id="doc_1", owner_id="user_1")

    assert ("FastAPI", "Python", "user_1") in driver.storage["relationships"]
    assert driver.storage["relationships"][("FastAPI", "Python", "user_1")]["doc_ids"] == ["doc_2"]
    assert ("FastAPI", "user_1") in driver.storage["nodes"]
    assert ("Python", "user_1") in driver.storage["nodes"]

    # Delete doc 2 -> relationship and orphan nodes are deleted
    manager.delete_document_graph(doc_id="doc_2", owner_id="user_1")

    assert ("FastAPI", "Python", "user_1") not in driver.storage["relationships"]
    assert ("FastAPI", "user_1") not in driver.storage["nodes"]
    assert ("Python", "user_1") not in driver.storage["nodes"]


@pytest.mark.asyncio
async def test_legacy_graph_data_invisible_and_purgeable():
    """Nodes without owner_id are legacy: invisible to retrieval and purged by purge_legacy_graph."""
    driver = FakeNeo4jDriver()
    manager = Neo4jManager(driver=driver)

    # Insert legacy node and relationship (no owner_id)
    driver.storage["nodes"][("LegacyEntity", "")] = {
        "name": "LegacyEntity",
        "owner_id": "",
        "doc_ids": [],
    }
    driver.storage["relationships"][("LegacyEntity", "LegacyTarget", "")] = {
        "source": "LegacyEntity",
        "target": "LegacyTarget",
        "owner_id": "",
        "doc_ids": [],
    }

    # Add owned entity for User A
    manager.add_entity("Entity", "ModernEntity", "Owned entity", owner_id="user_a", doc_id="doc_1")

    # Retrieval for User A or User B never returns legacy entity
    names_a = manager.get_all_entity_names(owner_id="user_a")
    assert "LegacyEntity" not in names_a
    assert "ModernEntity" in names_a

    with patch("app.rag.graph.graph_retriever.get_neo4j_manager", return_value=manager):
        ctx = await retrieve_graph_context("Tell me about LegacyEntity", owner_id="user_a")
        assert ctx == ""

    # Verify get_neo4j_manager returns manager when set
    with patch("app.rag.graph.neo4j_manager._manager", manager):
        assert get_neo4j_manager() is manager

    # Purge legacy graph via module helper
    with patch("app.rag.graph.neo4j_manager.get_neo4j_manager", return_value=manager):
        deleted_count = purge_legacy_graph()
        assert deleted_count == 1
        assert ("LegacyEntity", "") not in driver.storage["nodes"]
        assert ("ModernEntity", "user_a") in driver.storage["nodes"]

        # Second purge is idempotent
        assert purge_legacy_graph() == 0


@pytest.mark.asyncio
async def test_graph_extractor_ignores_malformed_llm_json():
    """R-48: extract_and_store_graph ignores malformed items (wrong types, missing keys) instead of raising."""
    driver = FakeNeo4jDriver()
    manager = Neo4jManager(driver=driver)

    malformed_payload = """{
        "entities": [
            {"name": "ValidNode", "type": "Concept", "description": "Good node"},
            {"invalid_key": 123},
            {"name": 456},
            "just a string",
            null
        ],
        "relationships": [
            {"source": "ValidNode", "target": "OtherNode", "type": "USES", "context": "Valid rel"},
            {"source": 123, "target": "OtherNode"},
            {"source": "ValidNode"},
            "bad rel",
            null
        ]
    }"""

    # Add OtherNode so relationship has endpoints
    manager.add_entity("Concept", "OtherNode", "Other", owner_id="user_1", doc_id="doc_1")

    with patch("app.rag.graph.graph_extractor.llm_manager.generate", new_callable=AsyncMock, return_value=malformed_payload), \
         patch("app.rag.graph.graph_extractor.get_neo4j_manager", return_value=manager):
        await extract_and_store_graph("some input text", owner_id="user_1", doc_id="doc_1")

    # Valid node should be stored
    assert ("ValidNode", "user_1") in driver.storage["nodes"]
    # Malformed entities should not cause an unhandled exception or corrupt nodes
    assert ("just a string", "user_1") not in driver.storage["nodes"]



@pytest.mark.asyncio
async def test_safe_delete_when_graph_disabled_or_error(monkeypatch):
    """Document delete/replace must be safe when DISABLE_GRAPH=1 or when Neo4j raises an error."""
    from app.api.routes.documents import delete_doc
    from unittest.mock import MagicMock
    from fastapi import Request

    # 1. DISABLE_GRAPH=1
    monkeypatch.setenv("DISABLE_GRAPH", "1")
    with patch("app.api.routes.documents.get_bm25_store") as mock_bm25, \
         patch("app.api.routes.documents.get_document_parents", return_value=[]), \
         patch("app.api.routes.documents.delete_document", new_callable=AsyncMock), \
         patch("app.api.routes.documents.bump_cache_generation", new_callable=AsyncMock), \
         patch("app.rag.graph.neo4j_manager.Neo4jManager.delete_document_graph") as mock_neo4j_del:
        mock_store = MagicMock()
        mock_store.get_document_metadata.return_value = [{"doc_id": "doc_123", "owner_id": "1"}]
        mock_bm25.return_value = mock_store

        mock_req = MagicMock(spec=Request)
        res = await delete_doc(doc_id="doc_123", request=mock_req, user_id=1)
        assert res == {"status": "deleted"}
        mock_neo4j_del.assert_not_called()

    # 2. Neo4j error -> logged as warning, delete does not fail
    monkeypatch.setenv("DISABLE_GRAPH", "0")
    with patch("app.api.routes.documents.get_bm25_store") as mock_bm25, \
         patch("app.api.routes.documents.get_document_parents", return_value=[]), \
         patch("app.api.routes.documents.delete_document", new_callable=AsyncMock), \
         patch("app.api.routes.documents.bump_cache_generation", new_callable=AsyncMock), \
         patch("app.rag.graph.neo4j_manager.get_neo4j_manager") as mock_get_mgr:
        mock_store = MagicMock()
        mock_store.get_document_metadata.return_value = [{"doc_id": "doc_123", "owner_id": "1"}]
        mock_bm25.return_value = mock_store

        failing_mgr = MagicMock()
        failing_mgr.delete_document_graph.side_effect = RuntimeError("Neo4j connection dropped")
        mock_get_mgr.return_value = failing_mgr

        mock_req = MagicMock(spec=Request)
        # Should not raise exception
        res = await delete_doc(doc_id="doc_123", request=mock_req, user_id=1)
        assert res == {"status": "deleted"}


def _check_neo4j_reachability() -> bool:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


requires_neo4j = pytest.mark.skipif(
    not _check_neo4j_reachability(),
    reason="Neo4j service is not reachable (Neo4j integration tests skipped)",
)


@requires_neo4j
def test_real_neo4j_integration_isolation():
    """Live Neo4j integration test (skipped when Neo4j is not reachable)."""
    manager = Neo4jManager()
    try:
        manager.add_entity("Project", "RealFalcon", "Confidential", owner_id="user_real_a", doc_id="doc_ra")
        names = manager.get_all_entity_names(owner_id="user_real_b")
        assert "RealFalcon" not in names

        names_a = manager.get_all_entity_names(owner_id="user_real_a")
        assert "RealFalcon" in names_a
    finally:
        manager.delete_document_graph("doc_ra", "user_real_a")
        manager.close()
