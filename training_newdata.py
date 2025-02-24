import torch
import os
import json
from typing import List
import logging
from tqdm import tqdm

def extend_knowledge_base(rag_system, additional_pdf_files: List[str], save_dir: str = 'saved_model'):
    """
    Extend an existing RAG system's knowledge base with additional PDF documents
    
    Args:
        rag_system: Initialized RAG system with existing knowledge base
        additional_pdf_files: List of new PDF files to add to the knowledge base
        save_dir: Directory where the updated knowledge base will be saved
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extending knowledge base with {len(additional_pdf_files)} new PDF documents")
    
    # Store original knowledge chunks and embeddings
    original_chunks = rag_system.knowledge_chunks.copy()
    original_embeddings = rag_system.chunk_embeddings.clone() if rag_system.chunk_embeddings is not None else None
    original_sources = rag_system.sources.copy()
    
    # Process new PDF files
    new_chunks = []
    
    for pdf_file in additional_pdf_files:
        chunks = rag_system.process_pdf(pdf_file)
        new_chunks.extend(chunks)
    
    if not new_chunks:
        logger.warning("No valid text chunks extracted from new PDFs")
        return False
    
    # Compute embeddings for new chunks
    logger.info(f"Computing embeddings for {len(new_chunks)} new knowledge chunks...")
    new_embeddings = rag_system.compute_embeddings(new_chunks)
    
    # Merge with existing knowledge base
    if original_embeddings is not None:
        rag_system.chunk_embeddings = torch.cat([original_embeddings, new_embeddings], dim=0)
    else:
        rag_system.chunk_embeddings = new_embeddings
    
    # Update knowledge chunks and sources
    rag_system.knowledge_chunks = original_chunks + new_chunks
    
    # Original sources dictionary is already updated in process_pdf
    
    # Save the extended knowledge base
    logger.info(f"Saving extended knowledge base to {save_dir}")
    rag_system.save(save_dir)
    
    logger.info(f"Knowledge base extended successfully with {len(new_chunks)} new chunks")
    return True

def main_extension():
    """Main function to extend an existing RAG knowledge base"""
    from base_training import RAGSystem  # Import your RAG class
    
    # Initialize RAG system
    rag = RAGSystem(use_cuda=torch.cuda.is_available())
    
    # Path to saved model
    save_dir = 'saved_model'
    
    # First load existing knowledge base
    try:
        rag.load(save_dir)
        print(f"Loaded existing knowledge base with {len(rag.knowledge_chunks)} chunks")
    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
        return
    
    # List of additional PDF files to incorporate
    additional_pdf_files = ['ts_training_data.pdf',]  # Update with your new PDF paths
    
    # Extend knowledge base
    success = extend_knowledge_base(rag, additional_pdf_files, save_dir)
    
    if success:
        print(f"Knowledge base extended successfully! Now contains {len(rag.knowledge_chunks)} total chunks")
        
        # Quick test to validate the extended knowledge base
        test_query = "Enter a test question here"
        result = rag.generate_response(test_query)
        
        if result['success']:
            print("\nTest response:", result['response'])
        else:
            print("\nError in test:", result['response'])
    else:
        print("Failed to extend knowledge base")

if __name__ == "__main__":
    main_extension()