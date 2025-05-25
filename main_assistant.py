#!/usr/bin/env python3
"""
Main Enhanced Fuzzy RAG Product Feature Assistant with Multi-Dictionary Support
"""

import logging
from typing import Dict, List, Any

from config import Config
from multi_dictionary_matcher import MultiDictionaryMatcher
from embeddings_manager import EmbeddingsManager
from llm_generator import LLMResponseGenerator

logger = logging.getLogger(__name__)

class EnhancedProductFeatureAssistant:
    """Main orchestrator for the Enhanced Fuzzy RAG Product Feature Assistant"""
    
    def __init__(
        self,
        dictionaries_path: str = None,
        llm_provider: str = None,
        llm_model: str = None,
        api_key: str = None
    ):
        """Initialize the Enhanced Product Feature Assistant"""
        self.dictionaries_path = dictionaries_path or Config.DICTIONARIES_PATH
        
        # Initialize components
        self.dictionary_matcher = None
        self.embeddings_manager = None
        self.llm_generator = None
        
        # Initialize LLM generator
        self.llm_generator = LLMResponseGenerator(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key
        )
        
        logger.info("Enhanced Product Feature Assistant initialized")
    
    def initialize_from_dictionaries(self) -> bool:
        """Initialize the system from the dictionaries"""
        try:
            # Ensure directories exist
            Config.ensure_directories()
            
            # Initialize multi-dictionary matcher
            logger.info("Initializing multi-dictionary matcher...")
            self.dictionary_matcher = MultiDictionaryMatcher(self.dictionaries_path)
            
            # Initialize embeddings manager with your specific settings
            logger.info("Initializing embeddings manager...")
            self.embeddings_manager = EmbeddingsManager(
                model_name="BAAI/bge-small-en",
                device="cpu",
                encode_kwargs={"normalize_embeddings": True},
                qdrant_url="http://localhost:6333",
                collection_name="vector_db"
            )
            
            # Create embeddings from all dictionary data
            all_dictionaries = self.dictionary_matcher.get_all_entities()
            if all_dictionaries:
                logger.info("Creating embeddings from all dictionary data...")
                self.embeddings_manager.add_multi_dictionary_embeddings(all_dictionaries)
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            return False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return a comprehensive response"""
        if not self._is_initialized():
            raise ValueError("System not initialized. Call initialize_from_dictionaries() first.")
        
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Extract all entities using multi-dictionary fuzzy matching
            identified_entities = self.dictionary_matcher.extract_entities(query)
            total_entities = sum(len(entities) for entities in identified_entities.values())
            logger.info(f"Identified {total_entities} entities across all types")
            
            # Step 2: Correct the query
            corrected_query = self.dictionary_matcher.correct_query(query, identified_entities)
            
            # Step 3: Get entity details and similar entities using embeddings
            entity_details = {}
            similar_entities = {}
            
            for entity_type, entities in identified_entities.items():
                if not entities:
                    continue
                    
                entity_details[entity_type] = {}
                similar_entities[entity_type] = {}
                
                for entity in entities:
                    entity_name = entity['name']
                    
                    # Get details for each identified entity
                    details = self.embeddings_manager.get_entity_details(entity_name, entity_type)
                    entity_details[entity_type][entity_name] = details
                    
                    # Get similar entities using vector search
                    similar = self.embeddings_manager.search_similar_entities(
                        query_text=entity_name,
                        entity_type=None,  # Search across all types for diversity
                        exclude_entity=entity_name
                    )
                    similar_entities[entity_type][entity_name] = similar
            
            # Step 4: Generate LLM response
            explanation = self.llm_generator.generate_multi_dictionary_response(
                original_query=query,
                corrected_query=corrected_query,
                identified_entities=identified_entities,
                entity_details=entity_details,
                similar_entities=similar_entities
            )
            
            # Step 5: Create the final response
            response = {
                'original_query': query,
                'corrected_query': corrected_query,
                'identified_entities': identified_entities,
                'entity_details': entity_details,
                'similar_entities': similar_entities,
                'explanation': explanation,
                'confidence': self._calculate_overall_confidence(identified_entities)
            }
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def _is_initialized(self) -> bool:
        """Check if the system is properly initialized"""
        return all([
            self.dictionary_matcher is not None,
            self.embeddings_manager is not None,
            self.llm_generator is not None
        ])
    
    def _calculate_overall_confidence(self, identified_entities: Dict[str, List[Dict]]) -> float:
        """Calculate an overall confidence score for the parsed query"""
        all_entities = []
        for entity_type, entities in identified_entities.items():
            all_entities.extend(entities)
        
        if not all_entities:
            return 0.0
            
        # Average the confidence scores of individual entities
        confidences = [entity.get('confidence', 0.0) for entity in all_entities]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Adjust based on the number of entities found and diversity
        entity_count_factor = min(1.0, len(all_entities) / 3)  # Scale up to 3 entities
        
        # Bonus for diversity across entity types
        entity_types_found = len([entities for entities in identified_entities.values() if entities])
        diversity_factor = min(1.0, entity_types_found / 2)  # Scale up to 2 types
        
        # Combine the factors
        overall_confidence = 0.6 * avg_confidence + 0.3 * entity_count_factor + 0.1 * diversity_factor
        
        return overall_confidence
    
    def get_available_entities(self) -> Dict[str, List[Dict]]:
        """Get all available entities from all dictionaries"""
        if not self.dictionary_matcher:
            return {}
        return self.dictionary_matcher.get_all_entities()
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict]:
        """Get all entities of a specific type"""
        if not self.dictionary_matcher:
            return []
        return self.dictionary_matcher.get_entities_by_type(entity_type)
    
    def search_entities(self, query: str, entity_type: str = None, limit: int = 5) -> List[Dict]:
        """Search for entities similar to a query"""
        if not self.embeddings_manager:
            return []
        return self.embeddings_manager.search_similar_entities(query, entity_type, limit)
    
    def add_custom_entity(self, entity_data: Dict[str, Any], entity_type: str) -> bool:
        """Add a custom entity to the system (for future enhancement)"""
        # This would require extending the dictionary matcher and embeddings manager
        # to handle dynamic entity addition
        logger.warning("Adding custom entities not yet implemented")
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the system"""
        status = {
            'initialized': self._is_initialized(),
            'dictionary_matcher_loaded': self.dictionary_matcher is not None,
            'embeddings_manager_loaded': self.embeddings_manager is not None,
            'llm_generator_loaded': self.llm_generator is not None,
            'llm_provider': self.llm_generator.provider if self.llm_generator else None,
        }
        
        if self.dictionary_matcher:
            all_entities = self.dictionary_matcher.get_all_entities()
            status['entity_counts'] = {
                entity_type: len(entities) 
                for entity_type, entities in all_entities.items()
            }
            status['total_entities'] = sum(status['entity_counts'].values())
        
        if self.embeddings_manager:
            embeddings_stats = self.embeddings_manager.get_stats()
            status.update({
                'vector_storage_type': embeddings_stats['vector_db_type'],
                'embeddings_count': embeddings_stats['embeddings_count'],
                'embedding_model': embeddings_stats['model_name'],
                'qdrant_connected': embeddings_stats['qdrant_connected']
            })
        
        return status


class LegacyProductFeatureAssistant:
    """Legacy implementation for backward compatibility"""
    
    def __init__(
        self,
        features_file: str = None,
        llm_provider: str = None,
        llm_model: str = None,
        api_key: str = None
    ):
        """Initialize the Legacy Product Feature Assistant"""
        self.features_file = features_file or Config.FEATURES_FILE
        
        # Initialize components
        self.fuzzy_matcher = None
        self.vector_retriever = None
        self.llm_generator = None
        
        logger.info("Legacy Product Feature Assistant initialized")
    
    def initialize_from_features_file(self) -> bool:
        """Initialize the system from the features file"""
        try:
            # Initialize fuzzy matcher
            logger.info("Initializing fuzzy matcher...")
            from fuzzy_matcher import FuzzyFeatureMatcher
            self.fuzzy_matcher = FuzzyFeatureMatcher(self.features_file)
            
            # Initialize vector retriever
            logger.info("Initializing vector retriever...")
            from vector_retriever import VectorRetriever
            self.vector_retriever = VectorRetriever(
                model_name="BAAI/bge-small-en",
                device="cpu",
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # Create feature vectors
            all_features = self.fuzzy_matcher.get_all_features()
            if all_features:
                logger.info("Creating feature vectors...")
                self.vector_retriever.add_features(all_features)
            
            # Initialize LLM generator
            self.llm_generator = LLMResponseGenerator()
            
            logger.info("Legacy system initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during legacy initialization: {e}")
            return False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return a comprehensive response - legacy method"""
        if not self._is_initialized():
            raise ValueError("System not initialized. Call initialize_from_features_file() first.")
        
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Extract features using fuzzy matching
            identified_features = self.fuzzy_matcher.extract_features(query)
            logger.info(f"Identified {len(identified_features)} features")
            
            # Step 2: Correct the query
            corrected_query = self.fuzzy_matcher.correct_query(query, identified_features)
            
            # Step 3: Get feature details and similar features using vector search
            feature_details = {}
            similar_features = {}
            
            for feature in identified_features:
                feature_name = feature['name']
                
                # Get details for each identified feature
                details = self.vector_retriever.get_feature_details(feature_name)
                feature_details[feature_name] = details
                
                # Get similar features using vector search
                similar = self.vector_retriever.search_similar_features(
                    query_text=feature_name,
                    exclude_feature=feature_name
                )
                similar_features[feature_name] = similar
            
            # Step 4: Generate LLM response
            explanation = self.llm_generator.generate_response(
                original_query=query,
                corrected_query=corrected_query,
                identified_features=identified_features,
                feature_details=feature_details,
                similar_features=similar_features
            )
            
            # Step 5: Create the final response
            response = {
                'original_query': query,
                'corrected_query': corrected_query,
                'identified_features': identified_features,
                'feature_details': feature_details,
                'similar_features': similar_features,
                'explanation': explanation,
                'confidence': self._calculate_overall_confidence(identified_features)
            }
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def _is_initialized(self) -> bool:
        """Check if the system is properly initialized"""
        return all([
            self.fuzzy_matcher is not None,
            self.vector_retriever is not None,
            self.llm_generator is not None
        ])
    
    def _calculate_overall_confidence(self, features: List[Dict]) -> float:
        """Calculate an overall confidence score for the parsed query"""
        if not features:
            return 0.0
            
        # Average the confidence scores of individual features
        confidences = [feature.get('confidence', 0.0) for feature in features]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Adjust based on the number of features found (more features = higher confidence)
        feature_count_factor = min(1.0, len(features) / 3)  # Scale up to 3 features
        
        # Combine the factors
        overall_confidence = 0.7 * avg_confidence + 0.3 * feature_count_factor
        
        return overall_confidence
    
    def get_available_features(self) -> List[Dict]:
        """Get all available features from the dictionary"""
        if not self.fuzzy_matcher:
            return []
        return self.fuzzy_matcher.get_all_features()
    
    def add_custom_feature(self, feature_data: Dict[str, Any]) -> bool:
        """Add a custom feature to the system (for future enhancement)"""
        # This would require extending the fuzzy matcher and vector retriever
        # to handle dynamic feature addition
        logger.warning("Adding custom features not yet implemented")
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the system"""
        status = {
            'initialized': self._is_initialized(),
            'fuzzy_matcher_loaded': self.fuzzy_matcher is not None,
            'vector_retriever_loaded': self.vector_retriever is not None,
            'llm_generator_loaded': self.llm_generator is not None,
            'llm_provider': self.llm_generator.provider if self.llm_generator else None,
            'feature_count': len(self.get_available_features()) if self.fuzzy_matcher else 0
        }
        
        if self.vector_retriever:
            status['vector_storage_type'] = self.vector_retriever.vector_db_type
            status['embeddings_count'] = len(self.vector_retriever.feature_vectors)
        
        return status