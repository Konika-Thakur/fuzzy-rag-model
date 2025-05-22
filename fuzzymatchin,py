#!/usr/bin/env python3
"""
Fuzzy matching module based on the existing FuzzyProductSearchParser implementation.
This module handles product feature recognition and correction with fuzzy matching.
"""

import os
import json
import logging
import re
import string
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict

import difflib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FuzzyMatcher:
    """
    Handles fuzzy matching of product features against feature dictionaries,
    building on the existing FuzzyProductSearchParser implementation.
    """
    
    def __init__(self, dictionaries_path: str):
        """
        Initialize the FuzzyMatcher
        
        Args:
            dictionaries_path: Path to the directory containing dictionary files
        """
        self.dictionaries_path = dictionaries_path
        
        # Load feature dictionary
        self.features_full, self.features_map = self._load_features_dictionary("featuresDictionary.json")
        logger.info(f"Loaded {len(self.features_full)} features from dictionary")
        
        # Build feature lists for matching
        self.canonical_features = list(self.features_map.values())
        self.all_feature_variations = list(self.features_map.keys())
        
        # Set of stop words to ignore during parsing
        self.stop_words = {
            "and", "or", "the", "a", "an", "by", "for", "from", "in", "of",
            "on", "to", "with", "about", "as", "at", "is"
        }
        
        # Build typo maps for features
        self._build_typo_maps()
    
    def _load_features_dictionary(self, filename: str) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Load features dictionary from a JSON file
        
        Args:
            filename: Name of the JSON file containing feature dictionary
            
        Returns:
            Tuple of (full_feature_data, feature_mapping)
            - full_feature_data: List of dictionaries with complete feature info
            - feature_mapping: Maps all variations to canonical feature names
        """
        filepath = os.path.join(self.dictionaries_path, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Dictionary file {filepath} not found, creating empty dictionaries")
            return [], {}
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                features_data = json.load(f)
                
            # Create mapping from all variations to canonical names
            feature_mapping = {}
            
            for feature in features_data:
                canonical_name = feature.get("feature", "")
                if not canonical_name:
                    continue
                    
                # Add canonical name to mapping (lowercase for case-insensitive matching)
                feature_mapping[canonical_name.lower()] = canonical_name
                
                # Add spelling variations
                for variation in feature.get("spelling_variations", []):
                    feature_mapping[variation.lower()] = canonical_name
                    
                # Add synonyms
                for synonym in feature.get("synonyms", []):
                    feature_mapping[synonym.lower()] = canonical_name
                    
            return features_data, feature_mapping
            
        except Exception as e:
            logger.error(f"Error loading feature dictionary {filename}: {str(e)}")
            return [], {}
    
    def _build_typo_maps(self) -> None:
        """
        Build maps for typo tolerance in feature matching
        """
        # Map for direct typo lookup
        self.feature_typo_map: Dict[str, str] = {}
        
        # Map for partial substring matching
        self.fuzzy_feature_map: Dict[str, List[str]] = defaultdict(list)
        
        # Build maps from all feature variations
        for variation, canonical in self.features_map.items():
            # Add partial substrings to fuzzy map (for auto-complete like behavior)
            for i in range(3, len(variation) + 1):
                substr = variation[:i].lower()
                self.fuzzy_feature_map[substr].append(canonical)
                
            # Generate common typos and map them to canonical names
            for typo in self._generate_typos(variation):
                self.feature_typo_map[typo.lower()] = canonical
    
    def _generate_typos(self, word: str) -> List[str]:
        """
        Generate simple typo variants for a word
        
        Args:
            word: The word to generate typos for
            
        Returns:
            List of common typo variants
        """
        typos = []
        
        # Remove vowels (common in fast typing)
        vowels = 'aeiou'
        for i, ch in enumerate(word):
            if ch.lower() in vowels:
                typo = word[:i] + word[i+1:]
                if len(typo) >= 2:
                    typos.append(typo)
                    
        # Remove spaces (common in compound terms)
        if ' ' in word:
            typos.append(word.replace(' ', ''))
            
        # Letter swap (common typing error)
        for i in range(len(word) - 1):
            typo = word[:i] + word[i+1] + word[i] + word[i+2:]
            typos.append(typo)
            
        # Common substitutions
        substitutions = {
            'a': ['e', 'q', 'w', 's', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            # add more as needed
        }
        
        for i, ch in enumerate(word):
            if ch.lower() in substitutions:
                for sub in substitutions[ch.lower()]:
                    typo = word[:i] + sub + word[i+1:]
                    typos.append(typo)
                    
        return typos
    
    def _fuzzy_match(self, term: str, candidates: List[str], threshold: float = 0.7) -> Tuple[Optional[str], float]:
        """
        Find the closest match among candidates using fuzzy matching
        
        Args:
            term: Term to match
            candidates: List of candidate strings to match against
            threshold: Minimum similarity score (0.0-1.0) to consider a match
            
        Returns:
            Tuple of (matched_term, similarity_score)
        """
        if not term or not candidates:
            return None, 0.0
            
        # First try exact match (case insensitive)
        for candidate in candidates:
            if term.lower() == candidate.lower():
                return candidate, 1.0
                
        # Then try direct typo map lookup
        if term.lower() in self.feature_typo_map:
            canonical = self.feature_typo_map[term.lower()]
            return canonical, 0.95  # High confidence for known typos
            
        # Next try fuzzy matching with difflib
        matches = difflib.get_close_matches(
            term.lower(), 
            [c.lower() for c in candidates], 
            n=1, 
            cutoff=threshold
        )
        
        if matches:
            # Find the original case version
            for candidate in candidates:
                if candidate.lower() == matches[0]:
                    # Calculate similarity score
                    similarity = difflib.SequenceMatcher(None, term.lower(), candidate.lower()).ratio()
                    return candidate, similarity
                    
        # Finally try substring matching for partial terms
        if len(term) >= 3:
            substr = term[:3].lower()
            candidates_from_substr = self.fuzzy_feature_map.get(substr, [])
            
            if len(set(candidates_from_substr)) == 1:
                # If all candidates point to the same feature, it's a match
                return candidates_from_substr[0], 0.8
                
        return None, 0.0
    
    def tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize a user query into individual tokens
        
        Args:
            query: Raw user query
            
        Returns:
            List of tokens
        """
        # Remove punctuation except hyphens
        trans = str.maketrans('', '', string.punctuation.replace('-', ''))
        clean_query = query.translate(trans)
        
        # Split into words and filter out stop words
        words = clean_query.split()
        tokens = [word for word in words if word.lower() not in self.stop_words]
        
        return tokens
    
    def extract_features(self, query: str) -> List[Dict]:
        """
        Extract all features from a query using fuzzy matching
        
        Args:
            query: User query
            
        Returns:
            List of identified features with metadata
        """
        identified_features = []
        tokens = self.tokenize_query(query)
        
        # First look for multi-word features in the original query
        for feature_data in self.features_full:
            canonical_name = feature_data.get("feature", "")
            if not canonical_name:
                continue
                
            # Check canonical name
            if canonical_name.lower() in query.lower():
                position = query.lower().find(canonical_name.lower())
                
                identified_features.append({
                    'name': canonical_name,
                    'original_text': query[position:position + len(canonical_name)],
                    'confidence': 1.0,
                    'position': {
                        'start': position,
                        'end': position + len(canonical_name)
                    }
                })
                continue
                
            # Check variations and synonyms
            variations = (
                feature_data.get("spelling_variations", []) + 
                feature_data.get("synonyms", [])
            )
            
            for variation in variations:
                if variation.lower() in query.lower():
                    position = query.lower().find(variation.lower())
                    
                    identified_features.append({
                        'name': canonical_name,  # Use canonical name
                        'original_text': query[position:position + len(variation)],
                        'confidence': 0.9,  # Slightly lower confidence for variations
                        'position': {
                            'start': position,
                            'end': position + len(variation)
                        }
                    })
                    break  # Found one variation, no need to check others
        
        # Create a mask of positions already matched
        matched_positions = set()
        for feature in identified_features:
            for i in range(feature['position']['start'], feature['position']['end']):
                matched_positions.add(i)
        
        # Look for remaining individual tokens
        for token in tokens:
            # Skip very short tokens
            if len(token) <= 2:
                continue
                
            # Try to find this token's position in the original query
            token_pos = query.lower().find(token.lower())
            
            # Check for overlap with already matched positions
            overlap = False
            if token_pos >= 0:
                for i in range(token_pos, token_pos + len(token)):
                    if i in matched_positions:
                        overlap = True
                        break
                        
            if overlap:
                continue
                
            # Try to match the token with fuzzy matching
            match, score = self._fuzzy_match(token, self.all_feature_variations, 0.7)
            
            if match and score >= 0.7:
                # Get the canonical feature name
                canonical = self.features_map.get(match.lower(), match)
                
                # Add to identified features
                position = {}
                if token_pos >= 0:
                    position = {
                        'start': token_pos,
                        'end': token_pos + len(token)
                    }
                    
                    # Mark positions as matched
                    for i in range(position['start'], position['end']):
                        matched_positions.add(i)
                        
                identified_features.append({
                    'name': canonical,
                    'original_text': token,
                    'confidence': score,
                    'position': position
                })
        
        return identified_features
    
    def correct_query(self, original_query: str, identified_features: List[Dict]) -> str:
        """
        Create a corrected version of the original query with canonical feature names
        
        Args:
            original_query: Original user query
            identified_features: List of identified features with their info
            
        Returns:
            Query with corrected feature terms
        """
        corrected_query = original_query
        
        # Sort features by their positions in reverse order (right to left)
        # This ensures replacements don't affect other positions
        sorted_features = sorted(
            identified_features,
            key=lambda x: x.get('position', {}).get('start', 0),
            reverse=True
        )
        
        # Replace each identified feature with its canonical name
        for feature in sorted_features:
            if 'position' in feature and 'original_text' in feature:
                start = feature['position']['start']
                end = feature['position']['end']
                corrected_query = (
                    corrected_query[:start] +
                    feature['name'] +
                    corrected_query[end:]
                )
        
        return corrected_query
    
    def get_all_features(self) -> List[Dict]:
        """
        Get all features from the feature dictionary
        
        Returns:
            List of all feature data
        """
        return self.features_full


if __name__ == "__main__":
    # Create sample features for testing
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
    os.makedirs("test_data", exist_ok=True)
    with open("test_data/featuresDictionary.json", "w") as f:
        json.dump(sample_features, f, indent=2)
    
    # Test the fuzzy matcher
    matcher = FuzzyMatcher("test_data")
    
    test_query = "Looking for a chair with highbak and metl legs"
    features = matcher.extract_features(test_query)
    
    print("Original query:", test_query)
    print("Identified features:")
    for feature in features:
        print(f"- {feature['name']} (confidence: {feature['confidence']:.2f})")
    
    corrected = matcher.correct_query(test_query, features)
    print("Corrected query:", corrected)