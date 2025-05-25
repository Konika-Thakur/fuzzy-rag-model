#!/usr/bin/env python3
"""
Enhanced Embeddings Manager with normalization support and multi-dictionary functionality
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union
import numpy as np

from config import Config

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logger.warning("SentenceTransformer not available. Install with: pip install sentence-transformers")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant client not available. Install with: pip install qdrant-client")

class EmbeddingsManager:
    """Enhanced embeddings manager with normalization and better Qdrant integration"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        encode_kwargs: Dict[str, Any] = None,
        qdrant_url: str = None,
        collection_name: str = None
    ):
        """
        Initialize the Embeddings Manager
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ('cpu' or 'cuda')
            encode_kwargs: Additional arguments for encoding (e.g., normalize_embeddings)
            qdrant_url: URL for the Qdrant vector database
            collection_name: Name of the collection in Qdrant
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = device or Config.DEVICE
        self.encode_kwargs = encode_kwargs or {"normalize_embeddings": Config.NORMALIZE_EMBEDDINGS}
        self.qdrant_url = qdrant_url or Config.QDRANT_URL
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_dim = Config.EMBEDDING_DIM
        
        # Initialize the embedding model
        self.model = None
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded embedding model: {self.model_name} on {self.device}")
                logger.info(f"Encode kwargs: {self.encode_kwargs}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                
        # Initialize vector storage
        self.qdrant_client = None
        self.vector_db_type = None
        self.feature_vectors = {}
        
        self._initialize_vector_storage()
    
    def _initialize_vector_storage(self):
        """Initialize vector storage (try Qdrant first, fall back to file-based)"""
        # Try Qdrant first
        if QDRANT_AVAILABLE:
            try:
                self.qdrant_client = QdrantClient(url=self.qdrant_url)
                
                # Check if collection exists
                collections = self.qdrant_client.get_collections()
                collection_exists = any(
                    collection.name == self.collection_name
                    for collection in collections.collections
                )
                
                if not collection_exists:
                    # Create collection with normalized vectors support
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=qdrant_models.VectorParams(
                            size=self.embedding_dim,
                            distance=qdrant_models.Distance.COSINE  # Works well with normalized embeddings
                        )
                    )
                    logger.info(f"Created new collection '{self.collection_name}' in Qdrant")
                
                self.vector_db_type = "qdrant"
                logger.info("Connected to Qdrant vector database")
                return
                
            except Exception as e:
                logger.warning(f"Failed to connect to Qdrant: {e}")
                self.qdrant_client = None
                
        # Fall back to file-based storage
        self.vector_db_type = "file"
        self._load_vectors_from_file()
        logger.info("Using file-based storage for vectors")
    
    def _load_vectors_from_file(self):
        """Load feature vectors from the fallback file"""
        Config.ensure_directories()
        filepath = os.path.join(Config.get_data_path(), Config.VECTOR_FALLBACK_FILE)
        
        if not os.path.exists(filepath):
            logger.info(f"Vector file {filepath} not found, will be created when needed")
            return
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.feature_vectors = json.load(f)
                
            logger.info(f"Loaded {len(self.feature_vectors)} feature vectors from file")
                
        except Exception as e:
            logger.warning(f"Failed to load feature vectors from file: {e}")
    
    def _save_vectors_to_file(self):
        """Save feature vectors to the fallback file"""
        if not self.feature_vectors:
            return
            
        Config.ensure_directories()
        filepath = os.path.join(Config.get_data_path(), Config.VECTOR_FALLBACK_FILE)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.feature_vectors, f, indent=2)
                
            logger.info(f"Saved {len(self.feature_vectors)} feature vectors to file")
        except Exception as e:
            logger.error(f"Failed to save feature vectors to file: {e}")
    
    def encode_text(self, text: Union[str, List[str]]) -> Optional[np.ndarray]:
        """
        Encode text into vector embedding(s) with normalization
        
        Args:
            text: Text string or list of strings to encode
            
        Returns:
            Vector embedding(s) as numpy array
        """
        if self.model is None:
            logger.warning("No embedding model available")
            return None
            
        try:
            # Use the encode_kwargs for normalization and other options
            embedding = self.model.encode(text, **self.encode_kwargs)
            
            # Ensure normalization if specified
            if self.encode_kwargs.get("normalize_embeddings", False):
                if isinstance(text, str):
                    # Single text input
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                else:
                    # Multiple text inputs
                    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
                    embedding = embedding / np.where(norms > 0, norms, 1)
                        
            return embedding
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None
    
    def add_multi_dictionary_embeddings(self, multi_dict_data: Dict[str, List[Dict[str, Any]]]) -> bool:
        """
        Create and store embeddings for multiple dictionary types
        
        Args:
            multi_dict_data: Dictionary containing data for each entity type
            
        Returns:
            Boolean indicating success
        """
        if not multi_dict_data or self.model is None:
            logger.warning("No dictionary data or embedding model available")
            return False
            
        total_entities = sum(len(entities) for entities in multi_dict_data.values())
        logger.info(f"Creating embeddings for {total_entities} entities across {len(multi_dict_data)} dictionary types")
        
        # Process each dictionary type
        for dict_type, entities_data in multi_dict_data.items():
            logger.info(f"Processing {len(entities_data)} {dict_type}...")
            
            # Prepare texts for batch encoding
            texts = []
            entity_names = []
            
            for entity in entities_data:
                entity_name = self._get_entity_name(entity, dict_type)
                if not entity_name:
                    continue
                
                # Create rich text for embedding based on entity type
                embedding_text = self._create_embedding_text(entity, dict_type)
                texts.append(embedding_text)
                entity_names.append(entity_name)
            
            if not texts:
                logger.warning(f"No valid {dict_type} to process")
                continue
            
            # Generate embeddings in batch for efficiency
            embeddings = self.encode_text(texts)
            if embeddings is None:
                logger.error(f"Failed to generate embeddings for {dict_type}")
                continue
            
            # Store embeddings
            for i, (entity_name, embedding) in enumerate(zip(entity_names, embeddings)):
                entity_data = entities_data[i]
                
                # Create vector data with entity type information
                vector_data = {
                    'entity_name': entity_name,
                    'entity_type': dict_type,
                    'description': entity_data.get('description', ''),
                    'vector': embedding.tolist(),
                    'spelling_variations': entity_data.get('spelling_variations', []),
                    'synonyms': entity_data.get('synonyms', [])
                }
                
                # Add type-specific fields
                if dict_type == 'features':
                    vector_data['benefits'] = entity_data.get('benefits', [])
                    vector_data['technical_specs'] = entity_data.get('technical_specs', {})
                elif dict_type == 'styles':
                    vector_data['characteristics'] = entity_data.get('characteristics', [])
                elif dict_type == 'products':
                    vector_data['subcategories'] = entity_data.get('subcategories', [])
                elif dict_type == 'places':
                    vector_data['typical_furniture'] = entity_data.get('typical_furniture', [])
                
                # Create unique key for multi-type storage
                storage_key = f"{dict_type}:{entity_name}"
                
                # Store in memory and vector DB
                self.feature_vectors[storage_key] = vector_data
                self._store_vector_in_db(storage_key, embedding, vector_data)
        
        # Save to fallback file
        self._save_vectors_to_file()
        
        logger.info(f"Created and stored embeddings for {len(self.feature_vectors)} entities")
        return True
    
    def add_features(self, features_data: List[Dict[str, Any]]) -> bool:
        """
        Create and store embeddings for multiple features (legacy support)
        
        Args:
            features_data: List of feature dictionaries
            
        Returns:
            Boolean indicating success
        """
        if not features_data or self.model is None:
            logger.warning("No feature data or embedding model available")
            return False
            
        logger.info(f"Creating embeddings for {len(features_data)} features")
        
        # Prepare texts for batch encoding
        texts = []
        feature_names = []
        
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
                
            texts.append(embedding_text)
            feature_names.append(feature_name)
        
        if not texts:
            logger.warning("No valid features to process")
            return False
        
        # Generate embeddings in batch for efficiency
        embeddings = self.encode_text(texts)
        if embeddings is None:
            logger.error("Failed to generate embeddings")
            return False
        
        # Store embeddings
        for i, (feature_name, embedding) in enumerate(zip(feature_names, embeddings)):
            feature_data = features_data[i]
            
            # Create vector data
            vector_data = {
                'feature_name': feature_name,
                'description': feature_data.get('description', ''),
                'vector': embedding.tolist(),
                'benefits': feature_data.get('benefits', []),
                'technical_specs': feature_data.get('technical_specs', {}),
                'spelling_variations': feature_data.get('spelling_variations', []),
                'synonyms': feature_data.get('synonyms', [])
            }
                
            # Store in memory and vector DB
            self.feature_vectors[feature_name] = vector_data
            self._store_vector_in_db(feature_name, embedding, vector_data)
            
        # Save to fallback file
        self._save_vectors_to_file()
            
        logger.info(f"Created and stored embeddings for {len(feature_names)} features")
        return True
    
    def _get_entity_name(self, entity: Dict[str, Any], dict_type: str) -> str:
        """Get the entity name based on dictionary type"""
        if dict_type == "features":
            return entity.get("feature", "")
        elif dict_type == "styles":
            return entity.get("name", "")
        elif dict_type == "products":
            return entity.get("type", "")
        elif dict_type == "places":
            return entity.get("name", "")
        return ""
    
    def _create_embedding_text(self, entity: Dict[str, Any], dict_type: str) -> str:
        """Create rich text for embedding based on entity type"""
        entity_name = self._get_entity_name(entity, dict_type)
        description = entity.get('description', '')
        variations = entity.get('spelling_variations', [])
        synonyms = entity.get('synonyms', [])
        
        # Start with basic info
        embedding_text = f"{dict_type} {entity_name}"
        if description:
            embedding_text += f" {description}"
        if variations:
            embedding_text += f" {' '.join(variations)}"
        if synonyms:
            embedding_text += f" {' '.join(synonyms)}"
        
        # Add type-specific information
        if dict_type == 'features':
            benefits = entity.get('benefits', [])
            if benefits:
                embedding_text += f" benefits: {' '.join(benefits)}"
        elif dict_type == 'styles':
            characteristics = entity.get('characteristics', [])
            if characteristics:
                embedding_text += f" characteristics: {' '.join(characteristics)}"
        elif dict_type == 'products':
            subcategories = entity.get('subcategories', [])
            if subcategories:
                embedding_text += f" types: {' '.join(subcategories)}"
        elif dict_type == 'places':
            typical_furniture = entity.get('typical_furniture', [])
            if typical_furniture:
                embedding_text += f" furniture: {' '.join(typical_furniture)}"
        
        return embedding_text
    
    def _store_vector_in_db(self, feature_name: str, embedding: np.ndarray, vector_data: Dict[str, Any]):
        """Store a vector in Qdrant"""
        if self.vector_db_type == "qdrant" and self.qdrant_client is not None:
            try:
                # Create point for Qdrant
                point = qdrant_models.PointStruct(
                    id=self._get_hash_id(feature_name),
                    vector=embedding.tolist(),
                    payload=vector_data
                )
                
                # Upsert the point
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )
                
                logger.debug(f"Stored vector for {feature_name} in Qdrant")
                
            except Exception as e:
                logger.warning(f"Failed to store vector in Qdrant: {e}")
    
    def search_similar_entities(
        self, 
        query_text: str, 
        entity_type: str = None,
        limit: int = None,
        exclude_entity: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for entities similar to the query text with multi-dictionary support
        
        Args:
            query_text: Text to search for
            entity_type: Filter by entity type (features, styles, products, places)
            limit: Maximum number of results to return
            exclude_entity: Entity name to exclude from results
            
        Returns:
            List of similar entities with similarity scores
        """
        limit = limit or Config.MAX_SIMILAR_FEATURES
        
        # Encode the query
        query_embedding = self.encode_text(query_text)
        if query_embedding is None:
            return self._get_default_similar_entities(exclude_entity or query_text, entity_type)
        
        # Search using Qdrant if available
        if self.vector_db_type == "qdrant" and self.qdrant_client is not None:
            return self._search_qdrant_multi(query_embedding, entity_type, limit, exclude_entity)
        else:
            return self._search_memory_multi(query_embedding, entity_type, limit, exclude_entity)
    
    def search_similar_features(
        self, 
        query_text: str, 
        limit: int = None,
        exclude_feature: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for features similar to the query text (legacy support)
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            exclude_feature: Feature name to exclude from results
            
        Returns:
            List of similar features with similarity scores
        """
        limit = limit or Config.MAX_SIMILAR_FEATURES
        
        # Encode the query
        query_embedding = self.encode_text(query_text)
        if query_embedding is None:
            return self._get_default_similar_features(exclude_feature or query_text)
        
        # Search using Qdrant if available
        if self.vector_db_type == "qdrant" and self.qdrant_client is not None:
            return self._search_qdrant(query_embedding, limit, exclude_feature)
        else:
            return self._search_memory(query_embedding, limit, exclude_feature)
    
    def _search_qdrant_multi(
        self, 
        query_embedding: np.ndarray, 
        entity_type: str = None,
        limit: int = 5,
        exclude_entity: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant with entity type filtering"""
        try:
            # Prepare search filter
            filter_conditions = []
            
            if entity_type:
                filter_conditions.append(
                    qdrant_models.FieldCondition(
                        key="entity_type",
                        match=qdrant_models.MatchValue(value=entity_type)
                    )
                )
            
            search_filter = None
            if filter_conditions:
                search_filter = qdrant_models.Filter(must=filter_conditions)
            
            if exclude_entity:
                exclude_filter = qdrant_models.Filter(
                    must_not=[
                        qdrant_models.FieldCondition(
                            key="entity_name",
                            match=qdrant_models.MatchValue(value=exclude_entity)
                        )
                    ]
                )
                
                if search_filter:
                    # Combine filters
                    search_filter = qdrant_models.Filter(
                        must=search_filter.must,
                        must_not=exclude_filter.must_not
                    )
                else:
                    search_filter = exclude_filter
            
            # Search in Qdrant
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=limit
            )
            
            similar_entities = []
            for result in results:
                similar_entities.append({
                    'name': result.payload.get('entity_name'),
                    'type': result.payload.get('entity_type'),
                    'description': result.payload.get('description', ''),
                    'similarity': result.score,
                    'benefits': result.payload.get('benefits', []),
                    'characteristics': result.payload.get('characteristics', []),
                    'subcategories': result.payload.get('subcategories', []),
                    'typical_furniture': result.payload.get('typical_furniture', [])
                })
                
            return similar_entities
            
        except Exception as e:
            logger.warning(f"Error searching Qdrant: {e}")
            return self._search_memory_multi(query_embedding, entity_type, limit, exclude_entity)
    
    def _search_memory_multi(
        self, 
        query_embedding: np.ndarray, 
        entity_type: str = None,
        limit: int = 5,
        exclude_entity: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar entities in memory using cosine similarity"""
        similarities = []
        
        # Calculate similarity with all entities in memory
        for storage_key, data in self.feature_vectors.items():
            # Parse storage key to get entity type and name
            if ':' in storage_key:
                stored_type, entity_name = storage_key.split(':', 1)
            else:
                # Backward compatibility for old storage format
                stored_type = 'features'
                entity_name = storage_key
            
            # Filter by entity type if specified
            if entity_type and stored_type != entity_type:
                continue
            
            # Skip excluded entity
            if exclude_entity and entity_name == exclude_entity:
                continue
            
            # Skip entities without vectors
            if 'vector' not in data:
                continue
            
            vector = np.array(data['vector'])
            
            # Calculate cosine similarity (embeddings are already normalized)
            if self.encode_kwargs.get("normalize_embeddings", False):
                similarity = np.dot(query_embedding, vector)
            else:
                similarity = np.dot(query_embedding, vector) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vector)
                )
            
            similarities.append({
                'name': entity_name,
                'type': stored_type,
                'description': data.get('description', ''),
                'similarity': float(similarity),
                'benefits': data.get('benefits', []),
                'characteristics': data.get('characteristics', []),
                'subcategories': data.get('subcategories', []),
                'typical_furniture': data.get('typical_furniture', [])
            })
        
        # Sort by similarity and take top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]
    
    def _search_qdrant(
        self, 
        query_embedding: np.ndarray, 
        limit: int, 
        exclude_feature: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant (legacy features only)"""
        try:
            # Prepare search filter
            search_filter = None
            if exclude_feature:
                search_filter = qdrant_models.Filter(
                    must_not=[
                        qdrant_models.FieldCondition(
                            key="feature_name",
                            match=qdrant_models.MatchValue(value=exclude_feature)
                        )
                    ]
                )
            
            # Search in Qdrant
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=limit
            )
            
            similar_features = []
            for result in results:
                similar_features.append({
                    'name': result.payload.get('feature_name'),
                    'description': result.payload.get('description', ''),
                    'similarity': result.score,
                    'benefits': result.payload.get('benefits', [])
                })
                
            return similar_features
            
        except Exception as e:
            logger.warning(f"Error searching Qdrant: {e}")
            return self._search_memory(query_embedding, limit, exclude_feature)
    
    def _search_memory(
        self, 
        query_embedding: np.ndarray, 
        limit: int, 
        exclude_feature: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar features in memory using cosine similarity (legacy)"""
        similarities = []
        
        # Calculate similarity with all features in memory
        for name, data in self.feature_vectors.items():
            # Skip excluded feature
            if exclude_feature and name == exclude_feature:
                continue
                
            # Skip features without vectors
            if 'vector' not in data:
                continue
                
            vector = np.array(data['vector'])
            
            # Calculate cosine similarity (embeddings are already normalized)
            if self.encode_kwargs.get("normalize_embeddings", False):
                similarity = np.dot(query_embedding, vector)
            else:
                similarity = np.dot(query_embedding, vector) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vector)
                )
            
            similarities.append({
                'name': name,
                'description': data.get('description', ''),
                'similarity': float(similarity),
                'benefits': data.get('benefits', [])
            })
        
        # Sort by similarity and take top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]
    
    def get_entity_details(self, entity_name: str, entity_type: str = None) -> Dict[str, Any]:
        """Get detailed information about a specific entity"""
        # Try different storage key formats
        possible_keys = []
        if entity_type:
            possible_keys.append(f"{entity_type}:{entity_name}")
        
        # Also try without type prefix for backward compatibility
        possible_keys.append(entity_name)
        
        # Try to retrieve from Qdrant first
        if self.vector_db_type == "qdrant" and self.qdrant_client is not None:
            try:
                filter_conditions = [
                    qdrant_models.FieldCondition(
                        key="entity_name",
                        match=qdrant_models.MatchValue(value=entity_name)
                    )
                ]
                
                if entity_type:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="entity_type",
                            match=qdrant_models.MatchValue(value=entity_type)
                        )
                    )
                
                results = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_models.Filter(must=filter_conditions),
                    limit=1
                )
                
                if results and results[0]:
                    return results[0][0].payload
            except Exception as e:
                logger.warning(f"Failed to retrieve entity details from Qdrant: {e}")
        
        # Fallback to in-memory dictionary
        for key in possible_keys:
            if key in self.feature_vectors:
                return self.feature_vectors[key]
        
        # Return default data if not found
        return {
            "entity_name": entity_name,
            "entity_type": entity_type or "unknown",
            "description": f"A {entity_name} that enhances the product",
            "benefits": [f"Improves functionality", f"Enhances user experience"],
            "technical_specs": {}
        }
    
    def get_feature_details(self, feature_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific feature (legacy support)"""
        # Try to retrieve from Qdrant first
        if self.vector_db_type == "qdrant" and self.qdrant_client is not None:
            try:
                results = self.qdrant_client.scroll(
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
        return {
            "feature_name": feature_name,
            "description": f"A {feature_name} feature that enhances the product",
            "benefits": [f"Improves functionality", f"Enhances user experience"],
            "technical_specs": {}
        }
    
    def _get_default_similar_entities(self, entity_name: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """Return default similar entities when search fails"""
        # Predefined similar entities for common cases
        default_similar = {
            "features": {
                "HighBack": [
                    {'name': 'Lumbar Support', 'type': 'features', 'description': 'Support for the lower back', 'similarity': 0.85},
                    {'name': 'Ergonomic Design', 'type': 'features', 'description': 'Designed for comfort and posture', 'similarity': 0.82}
                ],
                "Metal Legs": [
                    {'name': 'Industrial', 'type': 'styles', 'description': 'Modern industrial aesthetic', 'similarity': 0.88},
                    {'name': 'Modern', 'type': 'styles', 'description': 'Clean contemporary design', 'similarity': 0.84}
                ]
            },
            "styles": {
                "Modern": [
                    {'name': 'Scandinavian', 'type': 'styles', 'description': 'Clean design with natural materials', 'similarity': 0.85},
                    {'name': 'Industrial', 'type': 'styles', 'description': 'Raw materials and utility design', 'similarity': 0.75}
                ],
                "Industrial": [
                    {'name': 'Modern', 'type': 'styles', 'description': 'Contemporary minimalist design', 'similarity': 0.80},
                    {'name': 'Metal Legs', 'type': 'features', 'description': 'Durable metal construction', 'similarity': 0.85}
                ]
            },
            "products": {
                "Chair": [
                    {'name': 'Sofa', 'type': 'products', 'description': 'Upholstered seating for multiple people', 'similarity': 0.75},
                    {'name': 'Office', 'type': 'places', 'description': 'Workspace environment', 'similarity': 0.70}
                ]
            },
            "places": {
                "Office": [
                    {'name': 'Chair', 'type': 'products', 'description': 'Seating furniture', 'similarity': 0.80},
                    {'name': 'Desk', 'type': 'products', 'description': 'Work surface furniture', 'similarity': 0.85}
                ]
            }
        }
        
        if entity_type and entity_type in default_similar:
            if entity_name in default_similar[entity_type]:
                return default_similar[entity_type][entity_name]
        
        # Generic similar entities
        return [
            {'name': 'Modern', 'type': 'styles', 'description': 'Contemporary design style', 'similarity': 0.70},
            {'name': 'Ergonomic Design', 'type': 'features', 'description': 'Designed for comfort', 'similarity': 0.65}
        ]
    
    def _get_default_similar_features(self, feature_name: str) -> List[Dict[str, Any]]:
        """Return default similar features when search fails (legacy support)"""
        # Predefined similar features for common features
        default_similar = {
            "HighBack": [
                {'name': 'Lumbar Support', 'description': 'Support for the lower back', 'similarity': 0.85},
                {'name': 'Ergonomic Design', 'description': 'Designed for comfort and posture', 'similarity': 0.82}
            ],
            "Metal Legs": [
                {'name': 'Industrial Design', 'description': 'Modern industrial aesthetic', 'similarity': 0.88},
                {'name': 'Aluminum Frame', 'description': 'Lightweight but strong aluminum construction', 'similarity': 0.84}
            ],
            "Lumbar Support": [
                {'name': 'Ergonomic Design', 'description': 'Designed for comfort and posture', 'similarity': 0.90},
                {'name': 'HighBack', 'description': 'Chair back that extends above shoulder height', 'similarity': 0.85}
            ]
        }
        
        if feature_name in default_similar:
            return default_similar[feature_name]
            
        # Generic similar features
        return [
            {'name': 'Ergonomic Design', 'description': 'Designed for comfort and posture', 'similarity': 0.75},
            {'name': 'Premium Materials', 'description': 'High-quality construction materials', 'similarity': 0.70}
        ]
    
    def _get_hash_id(self, text: str) -> str:
        """Generate a hash ID for a string"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings manager"""
        stats = {
            'model_name': self.model_name,
            'device': self.device,
            'encode_kwargs': self.encode_kwargs,
            'vector_db_type': self.vector_db_type,
            'collection_name': self.collection_name,
            'embeddings_count': len(self.feature_vectors),
            'model_loaded': self.model is not None,
            'qdrant_connected': self.qdrant_client is not None
        }
        
        # Add Qdrant-specific stats if available
        if self.vector_db_type == "qdrant" and self.qdrant_client is not None:
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                stats['qdrant_points_count'] = collection_info.points_count
                stats['qdrant_vectors_size'] = collection_info.config.params.vectors.size
            except Exception as e:
                logger.warning(f"Failed to get Qdrant collection stats: {e}")
        
        return stats
    
    def clear_all_embeddings(self):
        """Clear all stored embeddings from memory and vector database"""
        try:
            # Clear memory storage
            self.feature_vectors.clear()
            
            # Clear Qdrant collection if available
            if self.vector_db_type == "qdrant" and self.qdrant_client is not None:
                try:
                    # Delete and recreate the collection
                    self.qdrant_client.delete_collection(self.collection_name)
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=qdrant_models.VectorParams(
                            size=self.embedding_dim,
                            distance=qdrant_models.Distance.COSINE
                        )
                    )
                    logger.info(f"Cleared and recreated Qdrant collection: {self.collection_name}")
                except Exception as e:
                    logger.warning(f"Failed to clear Qdrant collection: {e}")
            
            # Clear file storage
            self._save_vectors_to_file()
            
            logger.info("Cleared all embeddings from storage")
            
        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")
    
    def export_embeddings(self, filepath: str) -> bool:
        """Export all embeddings to a JSON file"""
        try:
            export_data = {
                'metadata': {
                    'model_name': self.model_name,
                    'device': self.device,
                    'encode_kwargs': self.encode_kwargs,
                    'embedding_dim': self.embedding_dim,
                    'export_timestamp': str(np.datetime64('now')),
                    'total_embeddings': len(self.feature_vectors)
                },
                'embeddings': self.feature_vectors
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(self.feature_vectors)} embeddings to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting embeddings: {e}")
            return False
    
    def import_embeddings(self, filepath: str) -> bool:
        """Import embeddings from a JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Validate import data structure
            if 'embeddings' not in import_data:
                logger.error("Invalid import file: missing 'embeddings' key")
                return False
            
            # Check metadata compatibility if available
            if 'metadata' in import_data:
                metadata = import_data['metadata']
                if metadata.get('model_name') != self.model_name:
                    logger.warning(f"Model mismatch: import={metadata.get('model_name')}, current={self.model_name}")
                if metadata.get('embedding_dim') != self.embedding_dim:
                    logger.warning(f"Dimension mismatch: import={metadata.get('embedding_dim')}, current={self.embedding_dim}")
            
            # Import embeddings
            imported_embeddings = import_data['embeddings']
            self.feature_vectors.update(imported_embeddings)
            
            # Store in vector database if available
            if self.vector_db_type == "qdrant" and self.qdrant_client is not None:
                points = []
                for storage_key, data in imported_embeddings.items():
                    if 'vector' in data:
                        point = qdrant_models.PointStruct(
                            id=self._get_hash_id(storage_key),
                            vector=data['vector'],
                            payload=data
                        )
                        points.append(point)
                
                if points:
                    try:
                        self.qdrant_client.upsert(
                            collection_name=self.collection_name,
                            points=points
                        )
                        logger.info(f"Imported {len(points)} embeddings to Qdrant")
                    except Exception as e:
                        logger.warning(f"Failed to import embeddings to Qdrant: {e}")
            
            # Save to file storage
            self._save_vectors_to_file()
            
            logger.info(f"Imported {len(imported_embeddings)} embeddings from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing embeddings: {e}")
            return False
    
    def get_entity_types(self) -> List[str]:
        """Get all available entity types in the current storage"""
        entity_types = set()
        
        for storage_key in self.feature_vectors.keys():
            if ':' in storage_key:
                entity_type = storage_key.split(':', 1)[0]
                entity_types.add(entity_type)
            else:
                # Legacy format assumes features
                entity_types.add('features')
        
        return list(entity_types)
    
    def get_entities_by_type(self, entity_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all entities of a specific type"""
        entities = {}
        
        for storage_key, data in self.feature_vectors.items():
            if ':' in storage_key:
                stored_type, entity_name = storage_key.split(':', 1)
                if stored_type == entity_type:
                    entities[entity_name] = data
            elif entity_type == 'features':
                # Legacy format for features
                entities[storage_key] = data
        
        return entities
    
    def update_entity_embedding(
        self, 
        entity_name: str, 
        entity_type: str, 
        new_data: Dict[str, Any]
    ) -> bool:
        """Update an existing entity's embedding with new data"""
        try:
            storage_key = f"{entity_type}:{entity_name}"
            
            # Check if entity exists
            if storage_key not in self.feature_vectors:
                logger.warning(f"Entity {entity_name} of type {entity_type} not found")
                return False
            
            # Update the data
            self.feature_vectors[storage_key].update(new_data)
            
            # Regenerate embedding if text content changed
            embedding_text = self._create_embedding_text(new_data, entity_type)
            new_embedding = self.encode_text(embedding_text)
            
            if new_embedding is not None:
                self.feature_vectors[storage_key]['vector'] = new_embedding.tolist()
                
                # Update in vector database
                self._store_vector_in_db(storage_key, new_embedding, self.feature_vectors[storage_key])
                
                # Save to file
                self._save_vectors_to_file()
                
                logger.info(f"Updated embedding for {entity_name} ({entity_type})")
                return True
            else:
                logger.error(f"Failed to generate new embedding for {entity_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating entity embedding: {e}")
            return False
    
    def delete_entity_embedding(self, entity_name: str, entity_type: str) -> bool:
        """Delete an entity's embedding from storage"""
        try:
            storage_key = f"{entity_type}:{entity_name}"
            
            # Remove from memory
            if storage_key in self.feature_vectors:
                del self.feature_vectors[storage_key]
            else:
                logger.warning(f"Entity {entity_name} of type {entity_type} not found in memory")
                return False
            
            # Remove from Qdrant if available
            if self.vector_db_type == "qdrant" and self.qdrant_client is not None:
                try:
                    point_id = self._get_hash_id(storage_key)
                    self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=qdrant_models.PointIdsList(points=[point_id])
                    )
                    logger.debug(f"Deleted {entity_name} from Qdrant")
                except Exception as e:
                    logger.warning(f"Failed to delete from Qdrant: {e}")
            
            # Save updated file storage
            self._save_vectors_to_file()
            
            logger.info(f"Deleted embedding for {entity_name} ({entity_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting entity embedding: {e}")
            return False
    
    def batch_update_embeddings(self, entities_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """Batch update embeddings for multiple entity types"""
        results = {'updated': 0, 'added': 0, 'errors': 0}
        
        for entity_type, entities in entities_data.items():
            logger.info(f"Batch updating {len(entities)} {entity_type} entities")
            
            for entity_data in entities:
                entity_name = self._get_entity_name(entity_data, entity_type)
                if not entity_name:
                    results['errors'] += 1
                    continue
                
                storage_key = f"{entity_type}:{entity_name}"
                
                try:
                    # Generate embedding
                    embedding_text = self._create_embedding_text(entity_data, entity_type)
                    embedding = self.encode_text(embedding_text)
                    
                    if embedding is None:
                        results['errors'] += 1
                        continue
                    
                    # Create vector data
                    vector_data = {
                        'entity_name': entity_name,
                        'entity_type': entity_type,
                        'description': entity_data.get('description', ''),
                        'vector': embedding.tolist(),
                        **entity_data
                    }
                    
                    # Check if updating or adding
                    if storage_key in self.feature_vectors:
                        results['updated'] += 1
                    else:
                        results['added'] += 1
                    
                    # Store the embedding
                    self.feature_vectors[storage_key] = vector_data
                    self._store_vector_in_db(storage_key, embedding, vector_data)
                    
                except Exception as e:
                    logger.error(f"Error processing {entity_name}: {e}")
                    results['errors'] += 1
        
        # Save to file
        self._save_vectors_to_file()
        
        logger.info(f"Batch update complete: {results}")
        return results