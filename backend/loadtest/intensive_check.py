import asyncio
import os
import structlog
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# Set mock env vars just in case
os.environ["CHROMA_PERSIST_DIR"] = "./chroma_data_test"
os.environ["MOCK_LLM"] = "false"

logger = structlog.get_logger(__name__)

async def run_intensive_test():
    logger.info("Starting Intensive RAG Pipeline Test...")
    
    try:
        # 1. Test Chunking
        logger.info("Testing Parent-Child Chunking...")
        from app.rag.chunkers.recursive_chunker import chunk_documents
        docs = [Document(page_content="This is a very long document about Acme Corp. Acme Corp is a technology company. They build anvils. John Smith is the CEO of Acme Corp. John Smith loves eating apples.", metadata={"filename": "test.txt"})]
        chunks = chunk_documents(docs)
        assert len(chunks) > 0, "Chunking failed to produce chunks"
        logger.info(f"Produced {len(chunks)} chunks.")
        
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        # 2. Test Local Embeddings
        logger.info("Testing Local Embeddings...")
        from app.rag.embeddings.local_embedder import embed_documents
        embeddings = await embed_documents(texts)
        assert len(embeddings) == len(texts), "Embeddings count mismatch"
        logger.info("Embeddings generated successfully.")
        
        # 3. Test Chroma Ingestion
        logger.info("Testing Chroma Dense Ingestion...")
        from app.rag.vectorstore.chroma_store import add_documents
        await add_documents(texts, embeddings, metadatas)
        logger.info("Chroma ingestion successful.")
        
        # 4. Test BM25 Ingestion
        logger.info("Testing BM25 Sparse Ingestion...")
        from app.rag.vectorstore.bm25_store import get_bm25_store
        bm25_docs = [{"id": metadatas[i].get("hash", f"test_{i}"), "text": texts[i]} for i in range(len(texts))]
        get_bm25_store().add_documents(bm25_docs)
        logger.info("BM25 ingestion successful.")
        
        # 5. Test Graph Extraction (OpenRouter)
        logger.info("Testing Graph Extraction (LLM Call)...")
        from app.rag.graph.graph_extractor import extract_and_store_graph
        parent_text = metadatas[0].get("parent_content", texts[0])
        await extract_and_store_graph(parent_text)
        logger.info("Graph extraction LLM call completed.")
        
        # 6. Test Hybrid Retrieval
        logger.info("Testing Hybrid Search + Reranking...")
        from app.rag.retrievers.hybrid_retriever import hybrid_search
        hybrid_results = await hybrid_search("Who is the CEO of Acme?", top_k=5)
        logger.info(f"Hybrid search returned {len(hybrid_results)} results.")
        
        # 7. Test Graph Retrieval (OpenRouter)
        logger.info("Testing Graph Retrieval (LLM Call)...")
        from app.rag.graph.graph_retriever import retrieve_graph_context
        graph_context = await retrieve_graph_context("Who is the CEO of Acme?")
        logger.info(f"Graph context retrieved length: {len(graph_context)}")
        
        # 8. Test Context Engineering
        logger.info("Testing Context Engineering...")
        from app.rag.context_engineering import prune_irrelevant_context, build_dynamic_prompt
        pruned = prune_irrelevant_context(hybrid_results, threshold=-10.0) # Low threshold for test
        build_dynamic_prompt(pruned, graph_context)
        logger.info("Prompt dynamically built successfully.")
        
        # 9. Test Full Pipeline Ask
        logger.info("Testing Full Pipeline Ask (End-to-End LLM Generation)...")
        from app.rag.pipeline import pipeline
        response = await pipeline.ask("Who is the CEO of Acme Corp?")
        logger.info(f"Pipeline Answer: {response.get('answer', '')[:100]}...")
        assert "answer" in response, "Pipeline response missing answer"
        
        logger.info("ALL INTENSIVE TESTS PASSED SUCCESSFULLY! ✅")
        
    except Exception as e:
        logger.error(f"INTENSIVE TEST FAILED: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_intensive_test())
