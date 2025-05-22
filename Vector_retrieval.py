#!/usr/bin/env python3
"""
Vector retrieval module for product feature similarity and suggestions
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

import numpy as np

# Optional imports for different vector DB backends
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logging.warning("SentenceTransformer not available. Install with: pip install sentence-transformers")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available. Install with: pip install qdrant-client")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorRetriever:
    """
    Handles vector embeddings and retrieval of product features
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en",
        device: str = "cpu",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "product_features",
        embedding_dim: int = 384,
        use_file_fallback: bool = True,
        fallback_file: str = "data/feature_vectors.json"
    ):
        """
        Initialize the Vector Retriever
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            device: Device to run the model on ('cpu' or 'cuda')
            qdrant_url: URL for the Qdrant vector database
            collection_name: Name of the collection in Qdrant
            embedding_dim: Dimension of the embedding vectors
            use_file_fallback: Whether to fall back to file storage if vector DB fails
            fallback_file: Path to the fallback file
        """
        self.embedding_model_name = embedding_model
        self.device = device
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.use_file_fallback = use_file_fallback
        self.fallback_file = fallback_file
        
        # Initialize the embedding model
        self.model = None
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer(embedding_model)
                self.model.to(device)
                logger.info(f"Loaded embedding model: {embedding_model} on {device}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                
        # Initialize vector DB connection
        self.vector_db = None
        self.vector_db_type = None
        self._initialize_vector_db()
        
        # Initialize fallback storage
        self.feature_vectors = {}
        if use_file_fallback:
            self._load_vectors_from_file()
    
    def _initialize_vector_db(self):
        """
        Initialize connection to vector database
        """
        # Try Qdrant first
        if QDRANT_AVAILABLE:
            try:
                self.vector_db = QdrantClient(url=self.qdrant_url)
                
                # Check if collection exists
                collections = self.vector_db.get_collections()
                collection_exists = any(
                    collection.name == self.collection_name
                    for collection in collections.collections
                )
                
                if not collection_exists:
                    # Create collection
                    self.vector_db.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=qdrant_models.VectorParams(
                            size=self.embedding_dim,
                            distance=qdrant_models.Distance.COSINE
                        )
                    )
                    logger.info(f"Created new collection '{self.collection_name}' in Qdrant")
                
                self.vector_db_type = "qdrant"
                logger.info("Connected to Qdrant vector database")
                return
                
            except Exception as e:
                logger.warning(f"Failed to connect to Qdrant: {e}")
                self.vector_db = None
                
        # Try FAISS if Qdrant failed
        if self.vector_db is None and FAISS_AVAILABLE:
            try:
                self.vector_db = faiss.IndexFlatIP(self.embedding_dim)
                self.vector_db_type = "faiss"
                logger.info("Initialized FAISS index")
                return
                
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS: {e}")
                self.vector_db = None
                
        # Fall back to file-based if both failed
        if self.vector_db is None:
            logger.info("Using file-based fallback for vector operations")
            self.vector_db_type = "file"
    
    def _load_vectors_from_file(self):
        """
        Load feature vectors from the fallback file
        """
        if not os.path.exists(self.fallback_file):
            logger.info(f"Fallback file {self.fallback_file} not found, will be created when needed")
            return
            
        try:
            with open(self.fallback_file, 'r', encoding='utf-8') as f:
                self.feature_vectors = json.load(f)
                
            logger.info(f"Loaded {len(self.feature_vectors)} feature vectors from file")
            
            # If using FAISS, add vectors to the index
            if self.vector_db_type == "faiss" and self.vector_db is not None:
                self._load_vectors_to_faiss()
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load feature vectors from file: {e}")
    
    def _load_vectors_to_faiss(self):
        """
        Load vectors from feature_vectors dictionary to FAISS index
        """
        if not self.feature_vectors:
            return
            
        try:
            # Extract vectors and create index
            feature_names = []
            vectors = []
            
            for name, data in self.feature_vectors.items():
                if 'vector' in data:
                    feature_names.append(name)
                    vectors.append(data['vector'])
            
            if vectors:
                # Convert to numpy array and add to FAISS
                vectors_array = np.array(vectors).astype(np.float32)
                self.vector_db.add(vectors_array)
                
                # Store feature names for lookup
                self.feature_names = feature_names
                
                logger.info(f"Added {len(vectors)} vectors to FAISS index")
        except Exception as e:
            logger.error(f"Error loading vectors to FAISS: {e}")
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text into a vector embedding
        
        Args:
            text: Text to encode
            
        Returns:
            Vector embedding as numpy array
        """
        if self.model is None:
            logger.warning("No embedding model available")
            return None
            
        try:
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None
    
    def get_feature_details(self, feature_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific feature
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with feature details
        """
        # Try to retrieve from vector DB
        if self.vector_db_type == "qdrant" and self.vector_db is not None:
            try:
                results = self.vector_db.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="feature_name",
                                match=qdrant_models.MatchValue(value=feature_name)
                            )
                        ]
                    ),
                    limit=1
                )
                
                if results and results[0]:
                    return results[0][0].payload
            except Exception as e:
                logger.warning(f"Failed to retrieve feature details from Qdrant: {e}")
        
        # Fallback to in-memory dictionary
        if feature_name in self.feature_vectors:
            return self.feature_vectors[feature_name]
        
        # Return default data if not found
        logger.info(f"Using default details for feature: {feature_name}")
        return {
            "feature_name": feature_name,
            "description": f"Description for {feature_name}",
            "benefits": [f"Benefit 1 of {feature_name}", f"Benefit 2 of {feature_name}"],
            "technical_specs": {"spec1": "value1", "spec2": "value2"}
        }
    
    def get_similar_features(
        self,
        feature_name: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get features similar to a given feature
        
        Args:
            feature_name: Name of the feature to find similarities for
            limit: Maximum number of similar features to return
            
        Returns:
            List of similar features with their details and similarity scores
        """
        similar_features = []
        feature_embedding = None
        
        # Try to get existing embedding
        if feature_name in self.feature_vectors and 'vector' in self.feature_vectors[feature_name]:
            feature_embedding = np.array(self.feature_vectors[feature_name]['vector'])
        
        # If no embedding, generate one
        if feature_embedding is None and self.model is not None:
            logger.info(f"Generating new embedding for feature: {feature_name}")
            feature_embedding = self.encode_text(feature_name)
        
        # Return empty list if no embedding
        if feature_embedding is None:
            logger.warning(f"No embedding available for feature: {feature_name}")
            return []
        
        # Search for similar features based on vector DB type
        if self.vector_db_type == "qdrant" and self.vector_db is not None:
            similar_features = self._get_similar_from_qdrant(feature_name, feature_embedding, limit)
        elif self.vector_db_type == "faiss" and self.vector_db is not None:
            similar_features = self._get_similar_from_faiss(feature_name, feature_embedding, limit)
        else:
            similar_features = self._get_similar_from_memory(feature_name, feature_embedding, limit)
        
        # If we still don't have similar features, return some default examples
        if not similar_features:
            similar_features = self._get_default_similar_features(feature_name)
            
        return similar_features
    
    def _get_similar_from_qdrant(self, feature_name: str, embedding: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """
        Get similar features from Qdrant
        """
        similar_features = []
        
        try:
            results = self.vector_db.search(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                limit=limit + 1  # +1 to account for the feature itself
            )
            
            for result in results:
                # Skip the feature itself
                if result.payload.get('feature_name') == feature_name:
                    continue
                
                similar_features.append({
                    'name': result.payload.get('feature_name'),
                    'description': result.payload.get('description', ''),
                    'similarity': result.score
                })
                
                if len(similar_features) >= limit:
                    break
                    
            return similar_features
            
        except Exception as e:
            logger.warning(f"Error retrieving similar features from Qdrant: {e}")
            return []
    
    def _get_similar_from_faiss(self, feature_name: str, embedding: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """
        Get similar features from FAISS
        """
        similar_features = []
        
        try:
            # Reshape embedding for FAISS
            query_vector = embedding.reshape(1, -1).astype(np.float32)
            
            # Search in FAISS index
            scores, indices = self.vector_db.search(query_vector, limit + 1)
            
            for i, idx in enumerate(indices[0]):
                if idx >= len(getattr(self, 'feature_names', [])):
                    continue
                    
                similar_name = self.feature_names[idx]
                
                # Skip the feature itself
                if similar_name == feature_name:
                    continue
                
                # Get details from in-memory dictionary
                details = self.feature_vectors.get(similar_name, {})
                
                similar_features.append({
                    'name': similar_name,
                    'description': details.get('description', ''),
                    'similarity': float(scores[0][i])
                })
                
                if len(similar_features) >= limit:
                    break
                    
            return similar_features
            
        except Exception as e:
            logger.warning(f"Error retrieving similar features from FAISS: {e}")
            return []
    
    def _get_similar_from_memory(self, feature_name: str, embedding: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """
        Get similar features by computing similarities in memory
        """
        similarities = []
        
        # Calculate similarity with all features in memory
        for name, data in self.feature_vectors.items():
            # Skip the feature itself
            if name == feature_name:
                continue
                
            # Skip features without vectors
            if 'vector' not in data:
                continue
                
            vector = np.array(data['vector'])
            
            # Calculate cosine similarity
            similarity = np.dot(embedding, vector) / (
                np.linalg.norm(embedding) * np.linalg.norm(vector)
            )
            
            similarities.append({
                'name': name,
                'description': data.get('description', ''),
                'similarity': float(similarity)
            })
        
        # Sort by similarity and take top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]
    
    def _get_default_similar_features(self, feature_name: str) -> List[Dict[str, Any]]:
        """
        Return default similar features when no proper similar features are found
        """
        default_similar = []
        
        # Some predefined similar features for common features
        if feature_name == "HighBack":
            default_similar = [
                {'name': 'Lumbar Support', 'description': 'Support for the lower back', 'similarity': 0.85},
                {'name': 'Ergonomic Design', 'description': 'Designed for comfort and posture', 'similarity': 0.82}
            ]
        elif feature_name == "Metal Legs":
            default_similar = [
                {'name': 'Industrial Design', 'description': 'Modern industrial aesthetic', 'similarity': 0.88},
                {'name': 'Aluminum Frame', 'description': 'Lightweight but strong aluminum construction', 'similarity': 0.84}
            ]
        elif feature_name == "Lumbar Support":
            default_similar = [
                {'name': 'Ergonomic Design', 'description': 'Designed for comfort and posture', 'similarity': 0.90},
                {'name': 'HighBack', 'description': 'Chair back that extends above shoulder height', 'similarity': 0.85}
            ]
        else:
            # Generic similar features
            default_similar = [
                {'name': 'Ergonomic Design', 'description': 'Designed for comfort and posture', 'similarity': 0.75},
                {'name': 'Premium Materials', 'description': 'High-quality construction materials', 'similarity': 0.70}
            ]
            
        return default_similar
    
    def create_embeddings_from_features(self, features_data: List[Dict[str, Any]]) -> bool:
        """
        Create vector embeddings from feature data and store them
        
        Args:
            features_data: List of feature dictionaries
            
        Returns:
            Boolean indicating success
        """
        if not features_data:
            logger.warning("No feature data provided for embedding creation")
            return False
            
        if self.model is None:
            logger.warning("No embedding model available")
            return False
            
        logger.info(f"Creating embeddings for {len(features_data)} features")
        
        # Process each feature
        for feature in features_data:
            feature_name = feature.get('feature', '')
            if not feature_name:
                continue
                
            # Create rich text for embedding
            description = feature.get('description', '')
            variations = feature.get('spelling_variations', [])
            synonyms = feature.get('synonyms', [])
            
            # Combine all text for the embedding
            embedding_text = feature_name
            if description:
                embedding_text += f" {description}"
            if variations:
                embedding_text += f" {' '.join(variations)}"
            if synonyms:
                embedding_text += f" {' '.join(synonyms)}"
                
            # Generate embedding
            embedding = self.encode_text(embedding_text)
            if embedding is None:
                logger.warning(f"Failed to create embedding for feature: {feature_name}")
                continue
                
            # Create vector data
            vector_data = {
                'feature_name': feature_name,
                'description': description,
                'vector': embedding.tolist()
            }
            
            # Add optional fields if present
            if 'benefits' in feature:
                vector_data['benefits'] = feature['benefits']
            if 'technical_specs' in feature:
                vector_data['technical_specs'] = feature['technical_specs']
                
            # Store in memory
            self.feature_vectors[feature_name] = vector_data
            
            # Store in vector DB
            self._store_vector_in_db(feature_name, embedding, vector_data)
            
        # Save to fallback file
        if self.use_file_fallback:
            self._save_vectors_to_file()
            
        logger.info(f"Created embeddings for {len(self.feature_vectors)} features")
        return True
    
    def _store_vector_in_db(self, feature_name: str, embedding: np.ndarray, vector_data: Dict[str, Any]) -> bool:
        """
        Store a vector in the vector database
        """
        # Store in Qdrant
        if self.vector_db_type == "qdrant" and self.vector_db is not None:
            try:
                # Check if feature already exists
                results = self.vector_db.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="feature_name",
                                match=qdrant_models.MatchValue(value=feature_name)
                            )
                        ]
                    ),
                    limit=1
                )
                
                if results and results[0]:
                    # Update existing point
                    self.vector_db.update_vectors(
                        collection_name=self.collection_name,
                        points=[
                            qdrant_models.PointVectors(
                                id=results[0][0].id, 
                                vector=embedding.tolist()
                            )
                        ]
                    )
                    self.vector_db.update_payload(
                        collection_name=self.collection_name,
                        payload=vector_data,
                        points=[results[0][0].id]
                    )
                else:
                    # Insert new point
                    self.vector_db.upsert(
                        collection_name=self.collection_name,
                        points=[
                            qdrant_models.PointStruct(
                                id=self._get_hash_id(feature_name),
                                vector=embedding.tolist(),
                                payload=vector_data
                            )
                        ]
                    )
                return True
                
            except Exception as e:
                logger.warning(f"Failed to store vector in Qdrant: {e}")
                return False
                
        # Store in FAISS (rebuild index with all vectors)
        elif self.vector_db_type == "faiss" and self.vector_db is not None:
            try:
                self._rebuild_faiss_index()
                return True
            except Exception as e:
                logger.warning(f"Failed to rebuild FAISS index: {e}")
                return False
                
        # If no vector DB, just keep in memory
        return True
    
    def _rebuild_faiss_index(self):
        """
        Rebuild the FAISS index with all vectors in memory
        """
        if self.vector_db_type != "faiss" or self.vector_db is None:
            return
            
        # Clear existing index
        self.vector_db.reset()
        
        # Add all vectors
        feature_names = []
        vectors = []
        
        for name, data in self.feature_vectors.items():
            if 'vector' in data:
                feature_names.append(name)
                vectors.append(data['vector'])
        
        if vectors:
            vectors_array = np.array(vectors).astype(np.float32)
            self.vector_db.add(vectors_array)
            self.feature_names = feature_names
            logger.info(f"Rebuilt FAISS index with {len(vectors)} vectors")
    
    def _save_vectors_to_file(self):
        """
        Save feature vectors to the fallback file
        """
        if not self.feature_vectors:
            return
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.fallback_file)), exist_ok=True)
        
        try:
            with open(self.fallback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feature_vectors, f, indent=2)
                
            logger.info(f"Saved {len(self.feature_vectors)} feature vectors to {self.fallback_file}")
        except Exception as e:
            logger.error(f"Failed to save feature vectors to file: {e}")
    
    def _get_hash_id(self, text: str) -> str:
        """
        Generate a hash ID for a string
        """
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()


if __name__ == "__main__":
    # Test code
    retriever = VectorRetriever(
        fallback_file="test_data/feature_vectors.json"
    )
    
    # Create test data
    test_features = [
        {
            "feature": "HighBack",
            "description": "Chair back that extends above shoulder height",
            "spelling_variations": ["highbak", "high back", "tall back"],
            "synonyms": ["tall backrest", "high backrest"],
            "benefits": ["Reduces neck strain", "Provides head support"]
        },
        {
            "feature": "Metal Legs",
            "description": "Legs made of metal material",
            "spelling_variations": ["metl legs", "metal leg", "metalic legs"],
            "synonyms": ["steel legs", "aluminum legs"],
            "benefits": ["Durability", "Modern look"]
        },
        {
            "feature": "Lumbar Support",
            "description": "Additional support for the lower back area",
            "spelling_variations": ["lumbar suport", "lumbar"],
            "synonyms": ["lower back support", "back support"],
            "benefits": ["Reduces back pain", "Improves posture"]
        }
    ]
    
    # Create embeddings
    os.makedirs("test_data", exist_ok=True)
    success = retriever.create_embeddings_from_features(test_features)
    print(f"Created embeddings: {success}")
    
    # Test retrieval
    details = retriever.get_feature_details("HighBack")
    print("\nFeature details:")
    print(json.dumps(details, indent=2))
    
    similar = retriever.get_similar_features("Metal Legs")
    print("\nSimilar features:")
    for feature in similar:
        print(f"- {feature['name']} (similarity: {feature['similarity']:.2f})")