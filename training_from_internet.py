import torch
import os
import json
import logging
from tqdm import tqdm
import wikipedia
import time
from typing import List, Dict, Any, Optional

def extend_with_wikipedia(
    rag_system, 
    topics: List[str], 
    max_articles_per_topic: int = 5,
    save_dir: str = 'saved_model',
    batch_size: int = 10
):
    """
    Extend an existing RAG system's knowledge base with Wikipedia articles
    
    Args:
        rag_system: Initialized RAG system with existing knowledge base
        topics: List of Wikipedia topics or search terms to retrieve
        max_articles_per_topic: Maximum number of articles to retrieve per topic
        save_dir: Directory where the updated knowledge base will be saved
        batch_size: Number of articles to process before updating embeddings
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extending knowledge base with Wikipedia articles on {len(topics)} topics")
    
    # Store original knowledge chunks and embeddings
    original_chunks = rag_system.knowledge_chunks.copy()
    original_embeddings = rag_system.chunk_embeddings.clone() if rag_system.chunk_embeddings is not None else None
    original_sources = rag_system.sources.copy() if hasattr(rag_system, 'sources') else {}
    
    # Track new files for batched processing
    new_chunks = []
    total_articles_processed = 0
    
    # Process topics in batches to avoid memory issues
    for topic_idx, topic in enumerate(topics):
        logger.info(f"Processing topic {topic_idx+1}/{len(topics)}: {topic}")
        
        try:
            # Search Wikipedia for the topic
            search_results = wikipedia.search(topic, results=max_articles_per_topic)
            
            for result in search_results:
                try:
                    # Get the Wikipedia page
                    wiki_page = wikipedia.page(result, auto_suggest=False)
                    
                    # Create a temporary text file with the Wikipedia content
                    temp_filename = f"temp_wiki_{total_articles_processed}.txt"
                    with open(temp_filename, 'w', encoding='utf-8') as f:
                        f.write(wiki_page.content)
                    
                    # Process the text file using the RAG system's text processing
                    article_chunks = process_wiki_article(rag_system, temp_filename, wiki_page.title, wiki_page.url)
                    new_chunks.extend(article_chunks)
                    
                    # Remove temporary file
                    os.remove(temp_filename)
                    
                    total_articles_processed += 1
                    logger.info(f"Added article: {wiki_page.title} ({len(article_chunks)} chunks)")
                    
                    # Process in batches to avoid memory issues
                    if len(new_chunks) >= batch_size:
                        # Update embeddings with current batch
                        update_embeddings_with_batch(rag_system, new_chunks, original_embeddings, original_chunks)
                        
                        # Reset batch tracking
                        original_chunks = rag_system.knowledge_chunks.copy()
                        original_embeddings = rag_system.chunk_embeddings.clone()
                        new_chunks = []
                        
                        # Save intermediate results
                        logger.info(f"Saving intermediate knowledge base with {len(rag_system.knowledge_chunks)} total chunks")
                        save_knowledge_base(rag_system, save_dir)
                
                except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                    logger.warning(f"Error retrieving Wikipedia article '{result}': {str(e)}")
                    continue
                
                # Rate limiting to avoid Wikipedia API throttling
                time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error processing topic '{topic}': {str(e)}")
            continue
    
    # Process any remaining chunks
    if new_chunks:
        update_embeddings_with_batch(rag_system, new_chunks, original_embeddings, original_chunks)
    
    # Save the final extended knowledge base
    logger.info(f"Saving extended knowledge base to {save_dir}")
    save_knowledge_base(rag_system, save_dir)
    
    logger.info(f"Knowledge base extended successfully with {total_articles_processed} Wikipedia articles")
    logger.info(f"Total knowledge chunks: {len(rag_system.knowledge_chunks)}")
    
    return True

def process_wiki_article(rag_system, file_path: str, title: str, url: str) -> List[str]:
    """Process a Wikipedia article and return chunks"""
    # Add source info to the RAG system
    if hasattr(rag_system, 'sources'):
        source_id = f"wiki_{len(rag_system.sources)}"
        rag_system.sources[source_id] = {
            "title": title,
            "type": "wikipedia",
            "url": url
        }
    
    # Use the RAG system's text processing methods
    # Note: Adjust based on your actual RAG implementation
    if hasattr(rag_system, 'process_text'):
        return rag_system.process_text(file_path, source_id=source_id)
    elif hasattr(rag_system, 'process_file'):
        return rag_system.process_file(file_path, source_id=source_id)
    else:
        # Fallback simple chunking if no specific method exists
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple chunk by paragraphs (adjust as needed)
        chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_with_metadata = {
                "text": chunk,
                "source": source_id,
                "chunk_id": f"{source_id}_chunk_{i}"
            }
            chunks_with_metadata.append(chunk_with_metadata)
        return chunks_with_metadata

def update_embeddings_with_batch(rag_system, new_chunks, original_embeddings, original_chunks):
    """Update the RAG system with a batch of new chunks"""
    logging.info(f"Computing embeddings for {len(new_chunks)} new knowledge chunks...")
    
    # Extract text from chunks if they're in dictionary format
    chunk_texts = []
    for chunk in new_chunks:
        if isinstance(chunk, dict) and "text" in chunk:
            chunk_texts.append(chunk["text"])
        else:
            chunk_texts.append(chunk)
    
    # Compute embeddings for new chunks
    new_embeddings = rag_system.compute_embeddings(chunk_texts)
    
    # Merge with existing knowledge base
    if original_embeddings is not None:
        rag_system.chunk_embeddings = torch.cat([original_embeddings, new_embeddings], dim=0)
    else:
        rag_system.chunk_embeddings = new_embeddings
    
    # Update knowledge chunks
    rag_system.knowledge_chunks = original_chunks + new_chunks

def save_knowledge_base(rag_system, save_dir: str):
    """Save the knowledge base to embeddings.pt and knowledge.json"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save embeddings
    torch.save(rag_system.chunk_embeddings, os.path.join(save_dir, 'embeddings.pt'))
    
    # Save knowledge chunks and sources
    knowledge_data = {
        "chunks": rag_system.knowledge_chunks,
        "sources": rag_system.sources if hasattr(rag_system, 'sources') else {}
    }
    
    with open(os.path.join(save_dir, 'knowledge.json'), 'w', encoding='utf-8') as f:
        json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Saved {len(rag_system.knowledge_chunks)} chunks to {save_dir}")

def load_knowledge_base(rag_system, save_dir: str):
    """Load knowledge base from embeddings.pt and knowledge.json"""
    embeddings_path = os.path.join(save_dir, 'embeddings.pt')
    knowledge_path = os.path.join(save_dir, 'knowledge.json')
    
    # Load embeddings
    if os.path.exists(embeddings_path):
        rag_system.chunk_embeddings = torch.load(embeddings_path)
    else:
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    # Load knowledge chunks and sources
    if os.path.exists(knowledge_path):
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
            
        rag_system.knowledge_chunks = knowledge_data.get("chunks", [])
        if hasattr(rag_system, 'sources'):
            rag_system.sources = knowledge_data.get("sources", {})
    else:
        raise FileNotFoundError(f"Knowledge file not found: {knowledge_path}")
    
    logging.info(f"Loaded {len(rag_system.knowledge_chunks)} chunks from {save_dir}")

def main_wikipedia_extension():
    """Main function to extend an existing RAG knowledge base with Wikipedia"""
    from base_training import RAGSystem  # Import your RAG class
    
    # Initialize RAG system
    rag = RAGSystem(use_cuda=torch.cuda.is_available())
    
    # Path to saved model
    save_dir = 'saved_model'
    
    # First load existing knowledge base
    try:
        load_knowledge_base(rag, save_dir)
        print(f"Loaded existing knowledge base with {len(rag.knowledge_chunks)} chunks")
    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
        return
    
    # List of Wikipedia topics to search for - customize based on your needs
    wikipedia_topics = [
        "Artificial intelligence",
        "Machine learning",
        "Natural language processing",
        "Transformer models",
        "Neural networks",
        "Deep learning",
        "Knowledge graphs",
        "Information retrieval",
        "Question answering systems",
        "Text embeddings",
        "Computer science",
        "Data structures",
        "Algorithms",
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "History",
        "Geography",
        "Economics"
        # Add more topics as needed
    ]
    
    # Extend knowledge base with Wikipedia articles
    success = extend_with_wikipedia(
        rag, 
        wikipedia_topics,
        max_articles_per_topic=5,  # Articles per topic
        save_dir=save_dir,
        batch_size=10  # Process in batches to avoid memory issues
    )
    
    if success:
        print(f"Knowledge base extended successfully! Now contains {len(rag.knowledge_chunks)} total chunks")
        
        # Quick test to validate the extended knowledge base
        test_queries = [
            "Explain how transformer models work",
            "What are the key concepts in machine learning?",
            "Tell me about neural networks"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: {query}")
            result = rag.generate_response(query)
            
            if result['success']:
                print("Response:", result['response'])
            else:
                print("Error:", result['response'])
    else:
        print("Failed to extend knowledge base")

if __name__ == "__main__":
    # Install required packages if needed
    # import subprocess
    # subprocess.check_call(["pip", "install", "wikipedia-api"])
    
    main_wikipedia_extension()