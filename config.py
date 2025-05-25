#!/usr/bin/env python3
"""
Centralized configuration for the Enhanced Fuzzy RAG Product Feature Assistant
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the Enhanced Product Feature Assistant"""
    
    # Directory paths
    DICTIONARIES_PATH = "dictionaries"
    DATA_PATH = "data"
    
    # Model configurations
    EMBEDDING_MODEL = "BAAI/bge-small-en"
    DEVICE = "cpu"  # Change to "cuda" if you have a GPU
    EMBEDDING_DIM = 384
    NORMALIZE_EMBEDDINGS = True
    
    # LLM configurations
    LLM_MODEL = "llama3.2:3b"
    LLM_PROVIDER = "ollama"  # Options: "ollama", "openai", "anthropic", "mock"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 512
    
    # Vector database configurations
    QDRANT_URL = "http://localhost:6333"
    COLLECTION_NAME = "vector_db"
    
    # Fuzzy matching parameters
    FUZZY_THRESHOLD = 0.7
    MAX_SIMILAR_FEATURES = 3
    MIN_TOKEN_LENGTH = 3
    
    # Stop words for query processing
    STOP_WORDS = {
        "and", "or", "the", "a", "an", "by", "for", "from", "in", "of",
        "on", "to", "with", "about", "as", "at", "is", "are", "was", "were"
    }
    
    # Dictionary file paths
    DICTIONARY_FILES = {
        "features": "featuresDictionary.json",
        "styles": "stylesDictionary.json", 
        "products": "productTypeDictionary.json",
        "places": "placesDictionary.json"
    }
    
    # Legacy file path for backward compatibility
    FEATURES_DICT_FILE = "featuresDictionary.json"
    VECTOR_FALLBACK_FILE = "feature_vectors.json"
    
    @classmethod
    def get_dictionaries_path(cls) -> str:
        """Get the full path to dictionaries directory"""
        return os.path.abspath(cls.DICTIONARIES_PATH)
    
    @classmethod
    def get_data_path(cls) -> str:
        """Get the full path to data directory"""
        return os.path.abspath(cls.DATA_PATH)
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist"""
        os.makedirs(cls.get_dictionaries_path(), exist_ok=True)
        os.makedirs(cls.get_data_path(), exist_ok=True)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "dictionaries_path": cls.DICTIONARIES_PATH,
            "embedding_model": cls.EMBEDDING_MODEL,
            "device": cls.DEVICE,
            "llm_model": cls.LLM_MODEL,
            "llm_provider": cls.LLM_PROVIDER,
            "qdrant_url": cls.QDRANT_URL,
            "collection_name": cls.COLLECTION_NAME,
            "fuzzy_threshold": cls.FUZZY_THRESHOLD,
            "max_similar_features": cls.MAX_SIMILAR_FEATURES
        }