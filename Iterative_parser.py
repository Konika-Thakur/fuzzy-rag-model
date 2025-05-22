#!/usr/bin/env python3
"""
Iterative parser module for extracting product features from user queries
Implements the parsing loop shown in the flowchart
"""

import logging
import re
from typing import Dict, List, Any, Set

from fuzzy_matching import FuzzyMatcher
from vector_retrieval import VectorRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IterativeParser:
    """
    Implements the iterative parsing loop for feature extraction from queries
    """
    
    def __init__(self, fuzzy_matcher: FuzzyMatcher, vector_retriever: VectorRetriever):
        """
        Initialize the Iterative Parser
        
        Args:
            fuzzy_matcher: FuzzyMatcher instance for feature matching
            vector_retriever: VectorRetriever instance for feature retrieval
        """
        self.fuzzy_matcher = fuzzy_matcher
        self.vector_retriever = vector_retriever
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the iterative parsing loop as shown in the flowchart
        
        Args:
            query: Raw user query
            
        Returns:
            Dictionary containing the parsed result
        """
        logger.info(f"Starting iterative parsing of query: {query}")
        
        original_query = query
        remaining_query = query
        all_identified_features = []
        processed_positions: Set[int] = set()
        iteration = 0
        max_iterations = 5  # Prevent infinite loops
        
        # Iterative parsing loop
        while remaining_query and iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}: Remaining query: {remaining_query}")
            
            # Step 1: Run Fuzzy Parser + RAG on current query
            features = self.fuzzy_matcher.extract_features(remaining_query)
            
            # Step 2: Attributes identified?
            if not features:
                logger.info("No additional features found, exiting loop")
                break  # Exit loop
            
            # Step 3: Extract identified attributes
            for feature in features:
                # Check if this feature position overlaps with already processed positions
                if 'position' in feature:
                    position = feature['position']
                    overlaps = False
                    
                    for i in range(position.get('start', 0), position.get('end', 0)):
                        if i in processed_positions:
                            overlaps = True
                            break
                    
                    if overlaps:
                        continue
                    
                    # Mark these positions as processed
                    for i in range(position.get('start', 0), position.get('end', 0)):
                        processed_positions.add(i)
                
                # Add the feature to our collection
                all_identified_features.append(feature)
            
            # Step 4: Remove identified terms from query
            modified_query = remaining_query
            for feature in features:
                if 'original_text' in feature and 'position' in feature:
                    original_text = feature['original_text']
                    # Replace with spaces to maintain character positions
                    modified_query = modified_query.replace(original_text, ' ' * len(original_text))
            
            # Step 5: Generate new substring query
            new_query = re.sub(r'\s+', ' ', modified_query).strip()
            
            # Step 6: Query changed?
            if new_query == remaining_query:
                logger.info("Query did not change, exiting loop")
                break  # Exit loop
            
            # Update remaining query for next iteration
            remaining_query = new_query
        
        # Step 7: Merge all identified attributes
        merged_features = self._merge_features(all_identified_features)
        
        # Generate corrected query
        corrected_query = self.fuzzy_matcher.correct_query(original_query, merged_features)
        
        # Calculate confidence scores
        for feature in merged_features:
            # If we have multiple occurrences of the same feature, increase confidence slightly
            if 'occurrences' in feature and feature['occurrences'] > 1:
                feature['confidence'] = min(1.0, feature['confidence'] * (1 + 0.1 * (feature['occurrences'] - 1)))
        
        logger.info(f"Corrected query: {corrected_query}")
        logger.info(f"Identified {len(merged_features)} features")
        
        return {
            'original_query': original_query,
            'corrected_query': corrected_query,
            'identified_features': merged_features,
            'confidence': self._calculate_overall_confidence(merged_features)
        }
    
    def _merge_features(self, features: List[Dict]) -> List[Dict]:
        """
        Merge multiple occurrences of the same feature
        
        Args:
            features: List of identified features
            
        Returns:
            List of merged features with duplicates combined
        """
        feature_map = {}
        
        for feature in features:
            name = feature['name']
            
            if name in feature_map:
                # Update the existing feature's confidence
                existing = feature_map[name]
                
                # Increment occurrence count
                existing['occurrences'] = existing.get('occurrences', 1) + 1
                
                # Keep the higher confidence
                if feature.get('confidence', 0) > existing.get('confidence', 0):
                    existing['confidence'] = feature['confidence']
                    
                    # If the new occurrence has position info, update it
                    if 'position' in feature and 'position' not in existing:
                        existing['position'] = feature['position']
                    
                    if 'original_text' in feature:
                        existing['original_text'] = feature['original_text']
            else:
                # Add occurrence count
                feature['occurrences'] = 1
                feature_map[name] = feature
        
        return list(feature_map.values())
    
    def _calculate_overall_confidence(self, features: List[Dict]) -> float:
        """
        Calculate an overall confidence score for the parsed query
        
        Args:
            features: List of identified features
            
        Returns:
            Overall confidence score (0.0-1.0)
        """
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


if __name__ == "__main__":
    import os
    import json
    from fuzzy_matching import FuzzyMatcher
    from vector_retrieval import VectorRetriever
    
    # Create test directories and data
    os.makedirs("test_data", exist_ok=True)
    
    # Create sample features
    sample_features = [
        {
            "feature": "HighBack",
            "description": "Chair back that extends above shoulder height",
            "spelling_variations": ["highbak", "high back", "tall back"],
            "synonyms": ["tall backrest", "high backrest"]
        },
        {
            "feature": "Metal Legs",
            "description": "Legs made of metal material",
            "spelling_variations": ["metl legs", "metal leg", "metalic legs"],
            "synonyms": ["steel legs", "aluminum legs"]
        }
    ]
    
    # Save sample features to file
    with open("test_data/featuresDictionary.json", "w") as f:
        json.dump(sample_features, f, indent=2)
    
    # Initialize components
    fuzzy_matcher = FuzzyMatcher("test_data")
    vector_retriever = VectorRetriever(fallback_file="test_data/feature_vectors.json")
    
    # Initialize test vectors
    vector_retriever.create_embeddings_from_features(sample_features)
    
    # Create parser
    parser = IterativeParser(fuzzy_matcher, vector_retriever)
    
    # Test the parser
    test_query = "Looking for a chair with highbak and metl legs"
    result = parser.parse(test_query)
    
    print("Iterative Parser Results:")
    print(f"Original query: {result['original_query']}")
    print(f"Corrected query: {result['corrected_query']}")
    print(f"Overall confidence: {result['confidence']:.2f}")
    print("\nIdentified features:")
    for feature in result['identified_features']:
        print(f"- {feature['name']} (confidence: {feature['confidence']:.2f})")