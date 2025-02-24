import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

class QuerySystem:
    def __init__(self, 
                 model_name='sentence-transformers/all-MiniLM-L6-v2',
                 knowledge_path='knowledge.json', 
                 embeddings_path='embeddings.pt'):
        self.device = "cpu"
        print(f"Using device: {self.device}")
        
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        print("Loading knowledge base...")
        self.load_knowledge_base(knowledge_path, embeddings_path)
        print("System initialized successfully!")

    def load_knowledge_base(self, knowledge_path, embeddings_path):
        try:
            if not os.path.exists(knowledge_path):
                raise FileNotFoundError(f"Knowledge base not found at {knowledge_path}")
                
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
                if isinstance(knowledge_data, dict):
                    self.knowledge_chunks = knowledge_data.get('chunks', [])
                else:
                    self.knowledge_chunks = knowledge_data
            
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
                
            self.chunk_embeddings = torch.load(embeddings_path, map_location=self.device)
            
            if len(self.knowledge_chunks) == 0:
                raise ValueError("Knowledge base is empty")
                
            print(f"Loaded {len(self.knowledge_chunks)} knowledge chunks")
            
        except Exception as e:
            raise Exception(f"Error loading knowledge base: {str(e)}")

    def get_answer(self, query):
        try:
            if not query or not query.strip():
                return {"success": False, "error": "Empty query"}

            encoded = self.tokenizer(
                query,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                query_embedding = outputs.last_hidden_state.mean(dim=1)
            
            similarities = cosine_similarity(
                query_embedding.cpu().numpy(),
                self.chunk_embeddings.cpu().numpy()
            )[0]
            
            best_idx = np.argmax(similarities)
            confidence = float(similarities[best_idx])
            
            # Confidence threshold
            if confidence < 0.3:
                return {
                    "success": True,
                    "answer": "I don't have enough confidence to provide an accurate answer to this question.",
                    "confidence": confidence
                }

            return {
                "success": True,
                "answer": self.knowledge_chunks[best_idx],
                "confidence": confidence
            }

        except Exception as e:
            return {"success": False, "error": f"Error processing query: {str(e)}"}

def main():
    try:
        knowledge_path = "saved_model/knowledge.json"
        embeddings_path = "saved_model/embeddings.pt"
        
        print("\nInitializing query system...")
        query_system = QuerySystem(
            knowledge_path=knowledge_path,
            embeddings_path=embeddings_path
        )
        
        print("\nSystem ready! Type 'quit' to exit.")
        
        while True:
            query = input("\nEnter your question: ").strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                print("\nPlease enter a valid question.")
                continue
                
            result = query_system.get_answer(query)
            
            if result["success"]:
                print(f"\nConfidence: {result['confidence']:.2f}")
                print(f"Answer: {result['answer']}")
            else:
                print(f"\nError: {result['error']}")
                
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        print("\nPlease ensure the saved_model directory contains valid knowledge.json and embeddings.pt files")

if __name__ == "__main__":
    main()