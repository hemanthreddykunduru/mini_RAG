import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
from PyPDF2 import PdfReader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging

class RAGSystem:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 use_cuda: bool = True,
                 max_length: int = 512,
                 chunk_size: int = 200):
        """
        Initialize RAG system with specified parameters
        
        Args:
            model_name: Name of the HuggingFace model to use
            use_cuda: Whether to use GPU if available
            max_length: Maximum sequence length for tokenizer
            chunk_size: Size of text chunks for knowledge base
        """
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.chunk_size = chunk_size
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.logger.info(f"Loading model {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Initialize knowledge base
        self.knowledge_chunks = []
        self.chunk_embeddings = None
        self.sources = {}  # Track source documents for chunks
        
    def process_pdf(self, pdf_path: str) -> List[str]:
        """
        Extract text chunks from PDF document
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text chunks
        """
        try:
            chunks = []
            source_name = os.path.basename(pdf_path)
            
            reader = PdfReader(pdf_path)
            full_text = ""
            
            # Extract text from all pages
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " "
                    
            # Clean text
            full_text = self._clean_text(full_text)
            
            # Split into chunks
            words = full_text.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word.split()) < self.chunk_size:
                    current_chunk.append(word)
                    current_length += len(word.split())
                else:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(chunk_text)
                        self.sources[chunk_text] = source_name
                    current_chunk = [word]
                    current_length = len(word.split())
            
            # Add final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                self.sources[chunk_text] = source_name
                
            self.logger.info(f"Extracted {len(chunks)} chunks from {source_name}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []
            
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = ''.join(char for char in text if char.isalnum() or char in ' .,?!-')
        return text
        
    def compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Compute embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of embeddings
        """
        embeddings = []
        
        for text in tqdm(texts, desc="Computing embeddings"):
            # Tokenize and encode text
            inputs = self.tokenizer(text, 
                                  max_length=self.max_length,
                                  padding=True,
                                  truncation=True,
                                  return_tensors='pt').to(self.device)
            
            # Compute embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding)
                
        return torch.cat(embeddings, dim=0)
        
    def build_knowledge_base(self, pdf_files: List[str]):
        """
        Build knowledge base from PDF files
        
        Args:
            pdf_files: List of paths to PDF files
        """
        self.logger.info("Building knowledge base...")
        self.knowledge_chunks = []
        
        # Process each PDF
        for pdf_file in pdf_files:
            chunks = self.process_pdf(pdf_file)
            self.knowledge_chunks.extend(chunks)
            
        if not self.knowledge_chunks:
            raise ValueError("No valid text chunks extracted from PDFs")
            
        # Compute embeddings
        self.logger.info("Computing embeddings for knowledge chunks...")
        self.chunk_embeddings = self.compute_embeddings(self.knowledge_chunks)
        
        self.logger.info(f"Knowledge base built with {len(self.knowledge_chunks)} chunks")
        
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: Query string
            k: Number of chunks to retrieve
            
        Returns:
            List of tuples (chunk_text, similarity_score, source)
        """
        # Compute query embedding
        query_embedding = self.compute_embeddings([query])
        
        # Compute similarities
        similarities = cosine_similarity(
            query_embedding.cpu().numpy(),
            self.chunk_embeddings.cpu().numpy()
        )[0]
        
        # Get top-k chunks with their scores and sources
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        
        for idx in top_k_indices:
            chunk = self.knowledge_chunks[idx]
            score = similarities[idx]
            source = self.sources.get(chunk, "Unknown")
            results.append((chunk, score, source))
            
        return results
        
    def generate_response(self, query: str) -> Dict:
        """
        Generate response for query
        
        Args:
            query: Query string
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Check if knowledge base exists
            if not self.knowledge_chunks or self.chunk_embeddings is None:
                return {
                    "response": "Knowledge base not initialized. Please build the knowledge base first.",
                    "success": False
                }
                
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query)
            
            if not relevant_chunks:
                return {
                    "response": "No relevant information found in the knowledge base.",
                    "success": False
                }
                
            # Format response
            response = f"Based on the available information:\n\n"
            for chunk, score, source in relevant_chunks:
                response += f"From {source} (confidence: {score:.2f}):\n{chunk}\n\n"
                
            return {
                "response": response,
                "success": True,
                "chunks": relevant_chunks
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "An error occurred while generating the response.",
                "success": False,
                "error": str(e)
            }
            
    def save(self, save_dir: str):
        """Save the knowledge base and embeddings"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save knowledge chunks and sources
        knowledge_data = {
            "chunks": self.knowledge_chunks,
            "sources": self.sources
        }
        
        with open(os.path.join(save_dir, 'knowledge.json'), 'w') as f:
            json.dump(knowledge_data, f)
            
        # Save embeddings
        if self.chunk_embeddings is not None:
            torch.save(self.chunk_embeddings, os.path.join(save_dir, 'embeddings.pt'))
            
        self.logger.info(f"Knowledge base saved to {save_dir}")
        
    def load(self, save_dir: str):
        """Load the knowledge base and embeddings"""
        # Load knowledge chunks and sources
        knowledge_path = os.path.join(save_dir, 'knowledge.json')
        if not os.path.exists(knowledge_path):
            raise FileNotFoundError(f"No knowledge base found at {knowledge_path}")
            
        with open(knowledge_path, 'r') as f:
            knowledge_data = json.load(f)
            
        self.knowledge_chunks = knowledge_data['chunks']
        self.sources = knowledge_data['sources']
        
        # Load embeddings
        embeddings_path = os.path.join(save_dir, 'embeddings.pt')
        if os.path.exists(embeddings_path):
            self.chunk_embeddings = torch.load(embeddings_path, map_location=self.device)
            
        self.logger.info(f"Knowledge base loaded from {save_dir}")

def main():
    # Initialize RAG system
    rag = RAGSystem(use_cuda=torch.cuda.is_available())
    
    # List of PDF files
    pdf_files = ['input.pdf']  # Update with your PDF paths
    
    # Build or load knowledge base
    save_dir = 'saved_model'
    
    if not os.path.exists(os.path.join(save_dir, 'knowledge.json')):
        try:
            rag.build_knowledge_base(pdf_files)
            rag.save(save_dir)
        except Exception as e:
            print(f"Error building knowledge base: {str(e)}")
            return
    else:
        try:
            rag.load(save_dir)
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            return
    
    print("\nRAG system ready! Type 'quit' to exit.")
    
    while True:
        query = input("\nEnter your question: ").strip()
        
        if query.lower() == 'quit':
            break
            
        try:
            result = rag.generate_response(query)
            
            if result['success']:
                print("\nResponse:", result['response'])
            else:
                print("\nError:", result['response'])
                
        except Exception as e:
            print(f"\nError processing query: {str(e)}")

if __name__ == "__main__":
    main()