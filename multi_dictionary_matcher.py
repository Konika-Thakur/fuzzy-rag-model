#!/usr/bin/env python3
"""
Enhanced fuzzy matching module for multiple dictionary types (features, styles, products, places)
"""

import os
import json
import logging
import re
import string
import difflib
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from config import Config

logger = logging.getLogger(__name__)

class MultiDictionaryMatcher:
    """Handles fuzzy matching across multiple dictionary types"""
    
    def __init__(self, dictionaries_path: str = None):
        """Initialize the Multi-Dictionary Matcher"""
        self.dictionaries_path = dictionaries_path or Config.DICTIONARIES_PATH
        
        # Storage for all dictionary types
        self.dictionaries = {}
        self.entity_maps = {}
        self.all_entities = {}
        
        # Load all dictionaries
        self._load_all_dictionaries()
        
        # Build combined entity lists for matching
        self._build_entity_maps()
        
        # Build typo maps for all entities
        self._build_typo_maps()
        
        logger.info(f"Loaded dictionaries: {list(self.dictionaries.keys())}")
    
    def _load_all_dictionaries(self):
        """Load all dictionary types from files"""
        for dict_type, filename in Config.DICTIONARY_FILES.items():
            filepath = os.path.join(self.dictionaries_path, filename)
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.dictionaries[dict_type] = data
                    logger.info(f"Loaded {len(data)} {dict_type} from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    self.dictionaries[dict_type] = []
            else:
                logger.warning(f"Dictionary file {filepath} not found")
                # Create sample data for missing dictionaries
                self.dictionaries[dict_type] = self._create_sample_dictionary(dict_type)
                self._save_sample_dictionary(dict_type, filepath)
    
    def _create_sample_dictionary(self, dict_type: str) -> List[Dict]:
        """Create sample dictionary data for a specific type"""
        if dict_type == "features":
            return [
                {
                    "feature": "HighBack",
                    "description": "Chair back that extends above shoulder height, providing support for the head and neck",
                    "spelling_variations": ["highbak", "high back", "tall back"],
                    "synonyms": ["tall backrest", "high backrest", "head rest"],
                    "benefits": ["Reduces neck strain", "Provides head support", "Improves posture"]
                },
                {
                    "feature": "Metal Legs",
                    "description": "Legs made of metal material such as steel or aluminum",
                    "spelling_variations": ["metl legs", "metal leg", "metalic legs"],
                    "synonyms": ["steel legs", "aluminum legs", "chrome legs"],
                    "benefits": ["Durability", "Modern look", "Stability"]
                },
                {
                    "feature": "Lumbar Support",
                    "description": "Additional support for the lower back area",
                    "spelling_variations": ["lumbar suport", "lumbar", "lower back suport"],
                    "synonyms": ["lower back support", "back support", "ergonomic back"],
                    "benefits": ["Reduces back pain", "Improves posture", "Prevents slouching"]
                }
            ]
        elif dict_type == "styles":
            return [
                {
                    "name": "Modern",
                    "description": "Clean lines and minimalist aesthetic with contemporary design elements",
                    "spelling_variations": ["modernist", "contemporary"],
                    "synonyms": ["contemporary", "current", "minimalist"],
                    "characteristics": ["Clean lines", "Neutral colors", "Functional design"]
                },
                {
                    "name": "Industrial",
                    "description": "Raw materials and utilitarian design inspired by factories and warehouses",
                    "spelling_variations": ["industrial style", "factory style"],
                    "synonyms": ["factory", "warehouse", "utility", "urban loft"],
                    "characteristics": ["Metal materials", "Exposed elements", "Raw finishes"]
                },
                {
                    "name": "Scandinavian",
                    "description": "Simple, clean design with natural materials and light colors",
                    "spelling_variations": ["nordic", "scandi"],
                    "synonyms": ["Nordic", "Danish", "Swedish"],
                    "characteristics": ["Light wood", "White/light colors", "Simple forms"]
                }
            ]
        elif dict_type == "products":
            return [
                {
                    "type": "Chair",
                    "description": "A seat with a back and often with arms, designed for one person",
                    "spelling_variations": ["chairs", "seat"],
                    "synonyms": ["seating", "seat", "armchair"],
                    "subcategories": ["Office chair", "Dining chair", "Lounge chair"]
                },
                {
                    "type": "Desk",
                    "description": "A piece of furniture with a flat surface for writing, working, or using a computer",
                    "spelling_variations": ["desks", "table"],
                    "synonyms": ["workstation", "table", "work desk"],
                    "subcategories": ["Office desk", "Writing desk", "Standing desk"]
                },
                {
                    "type": "Sofa",
                    "description": "A long upholstered seat with a back and arms, for two or more people",
                    "spelling_variations": ["sofas", "couch"],
                    "synonyms": ["couch", "settee", "loveseat"],
                    "subcategories": ["Sectional sofa", "Loveseat", "Sleeper sofa"]
                }
            ]
        elif dict_type == "places":
            return [
                {
                    "name": "Living Room",
                    "description": "The main room in a home where family and guests gather for relaxation",
                    "spelling_variations": ["livingroom", "living area"],
                    "synonyms": ["lounge", "family room", "sitting room"],
                    "typical_furniture": ["Sofa", "Coffee table", "TV stand"]
                },
                {
                    "name": "Office",
                    "description": "A room or building where business, professional, or clerical work is conducted",
                    "spelling_variations": ["office space", "workplace"],
                    "synonyms": ["workspace", "study", "workplace"],
                    "typical_furniture": ["Desk", "Office chair", "Filing cabinet"]
                },
                {
                    "name": "Bedroom",
                    "description": "A room furnished with a bed and intended primarily for sleeping",
                    "spelling_variations": ["bed room", "sleeping room"],
                    "synonyms": ["sleeping quarters", "bedchamber"],
                    "typical_furniture": ["Bed", "Dresser", "Nightstand"]
                }
            ]
        else:
            return []
    
    def _save_sample_dictionary(self, dict_type: str, filepath: str):
        """Save sample dictionary to file"""
        try:
            Config.ensure_directories()
            data = self.dictionaries[dict_type]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Created sample {dict_type} dictionary at {filepath}")
        except Exception as e:
            logger.error(f"Failed to save sample {dict_type} dictionary: {e}")
    
    def _build_entity_maps(self):
        """Build mapping from all variations to canonical names for each dictionary type"""
        for dict_type, data in self.dictionaries.items():
            entity_map = {}
            
            for item in data:
                # Get the canonical name based on dictionary type
                canonical_name = self._get_canonical_name(item, dict_type)
                if not canonical_name:
                    continue
                
                # Add canonical name to mapping (lowercase for case-insensitive matching)
                entity_map[canonical_name.lower()] = {
                    'canonical': canonical_name,
                    'type': dict_type,
                    'data': item
                }
                
                # Add spelling variations
                for variation in item.get("spelling_variations", []):
                    entity_map[variation.lower()] = {
                        'canonical': canonical_name,
                        'type': dict_type,
                        'data': item
                    }
                
                # Add synonyms
                for synonym in item.get("synonyms", []):
                    entity_map[synonym.lower()] = {
                        'canonical': canonical_name,
                        'type': dict_type,
                        'data': item
                    }
                
                # Add subcategories for products
                if dict_type == "products":
                    for subcategory in item.get("subcategories", []):
                        entity_map[subcategory.lower()] = {
                            'canonical': canonical_name,
                            'type': dict_type,
                            'data': item
                        }
            
            self.entity_maps[dict_type] = entity_map
            
        # Build combined entity map for general searching
        self.all_entities = {}
        for dict_type, entity_map in self.entity_maps.items():
            self.all_entities.update(entity_map)
    
    def _get_canonical_name(self, item: Dict, dict_type: str) -> str:
        """Get the canonical name field based on dictionary type"""
        if dict_type == "features":
            return item.get("feature", "")
        elif dict_type == "styles":
            return item.get("name", "")
        elif dict_type == "products":
            return item.get("type", "")
        elif dict_type == "places":
            return item.get("name", "")
        return ""
    
    def _build_typo_maps(self):
        """Build maps for typo tolerance across all entity types"""
        self.entity_typo_map = {}
        self.fuzzy_entity_map = defaultdict(list)
        
        # Build maps from all entity variations
        for entity_text, entity_info in self.all_entities.items():
            canonical = entity_info['canonical']
            
            # Add partial substrings to fuzzy map
            for i in range(3, len(entity_text) + 1):
                substr = entity_text[:i].lower()
                self.fuzzy_entity_map[substr].append(entity_info)
            
            # Generate common typos
            for typo in self._generate_typos(entity_text):
                self.entity_typo_map[typo.lower()] = entity_info
    
    def _generate_typos(self, word: str) -> List[str]:
        """Generate simple typo variants for a word"""
        typos = []
        
        # Remove vowels (common in fast typing)
        vowels = 'aeiou'
        for i, ch in enumerate(word):
            if ch.lower() in vowels and len(word) > 3:
                typo = word[:i] + word[i+1:]
                typos.append(typo)
        
        # Remove spaces (common in compound terms)
        if ' ' in word:
            typos.append(word.replace(' ', ''))
        
        # Letter swap (common typing error)
        for i in range(len(word) - 1):
            typo = word[:i] + word[i+1] + word[i] + word[i+2:]
            typos.append(typo)
        
        return typos
    
    def _fuzzy_match(self, term: str, candidates: Dict, threshold: float = None) -> Tuple[Optional[Dict], float]:
        """Find the closest match among candidates using fuzzy matching"""
        threshold = threshold or Config.FUZZY_THRESHOLD
        
        if not term or not candidates:
            return None, 0.0
        
        # First try exact match (case insensitive)
        if term.lower() in candidates:
            return candidates[term.lower()], 1.0
        
        # Then try direct typo map lookup
        if term.lower() in self.entity_typo_map:
            return self.entity_typo_map[term.lower()], 0.95
        
        # Next try fuzzy matching with difflib
        candidate_keys = list(candidates.keys())
        matches = difflib.get_close_matches(
            term.lower(),
            candidate_keys,
            n=1,
            cutoff=threshold
        )
        
        if matches:
            matched_key = matches[0]
            similarity = difflib.SequenceMatcher(None, term.lower(), matched_key).ratio()
            return candidates[matched_key], similarity
        
        # Finally try substring matching for partial terms
        if len(term) >= Config.MIN_TOKEN_LENGTH:
            substr = term[:3].lower()
            candidates_from_substr = self.fuzzy_entity_map.get(substr, [])
            
            if len(set(info['canonical'] for info in candidates_from_substr)) == 1:
                return candidates_from_substr[0], 0.8
        
        return None, 0.0
    
    def tokenize_query(self, query: str) -> List[str]:
        """Tokenize a user query into individual tokens"""
        # Remove punctuation except hyphens
        trans = str.maketrans('', '', string.punctuation.replace('-', ''))
        clean_query = query.translate(trans)
        
        # Split into words and filter out stop words
        words = clean_query.split()
        tokens = [word for word in words if word.lower() not in Config.STOP_WORDS]
        
        return tokens
    
    def extract_entities(self, query: str) -> Dict[str, List[Dict]]:
        """Extract all entities from a query using fuzzy matching, grouped by type"""
        identified_entities = {
            'features': [],
            'styles': [],
            'products': [],
            'places': []
        }
        
        tokens = self.tokenize_query(query)
        matched_positions = set()
        
        # First look for multi-word entities in the original query
        for dict_type, entity_map in self.entity_maps.items():
            for entity_text, entity_info in entity_map.items():
                if ' ' in entity_text and entity_text in query.lower():
                    position = query.lower().find(entity_text)
                    
                    # Check if this position is already matched
                    overlap = any(i in matched_positions for i in range(position, position + len(entity_text)))
                    if overlap:
                        continue
                    
                    identified_entities[dict_type].append({
                        'name': entity_info['canonical'],
                        'original_text': query[position:position + len(entity_text)],
                        'confidence': 0.95,
                        'type': dict_type,
                        'position': {
                            'start': position,
                            'end': position + len(entity_text)
                        },
                        'data': entity_info['data']
                    })
                    
                    # Mark positions as matched
                    for i in range(position, position + len(entity_text)):
                        matched_positions.add(i)
        
        # Then look for individual tokens
        for token in tokens:
            if len(token) <= Config.MIN_TOKEN_LENGTH - 1:
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
            match_info, score = self._fuzzy_match(token, self.all_entities)
            
            if match_info and score >= Config.FUZZY_THRESHOLD:
                entity_type = match_info['type']
                
                position = {}
                if token_pos >= 0:
                    position = {
                        'start': token_pos,
                        'end': token_pos + len(token)
                    }
                    
                    # Mark positions as matched
                    for i in range(position['start'], position['end']):
                        matched_positions.add(i)
                
                identified_entities[entity_type].append({
                    'name': match_info['canonical'],
                    'original_text': token,
                    'confidence': score,
                    'type': entity_type,
                    'position': position,
                    'data': match_info['data']
                })
        
        return identified_entities
    
    def correct_query(self, original_query: str, identified_entities: Dict[str, List[Dict]]) -> str:
        """Create a corrected version of the original query with canonical entity names"""
        corrected_query = original_query
        
        # Flatten all entities and sort by position
        all_entities = []
        for entity_type, entities in identified_entities.items():
            all_entities.extend(entities)
        
        # Sort by position in reverse order (right to left)
        sorted_entities = sorted(
            all_entities,
            key=lambda x: x.get('position', {}).get('start', 0),
            reverse=True
        )
        
        # Replace each identified entity with its canonical name
        for entity in sorted_entities:
            if 'position' in entity and 'start' in entity['position']:
                start = entity['position']['start']
                end = entity['position']['end']
                corrected_query = (
                    corrected_query[:start] +
                    entity['name'] +
                    corrected_query[end:]
                )
        
        return corrected_query
    
    def get_all_entities(self) -> Dict[str, List[Dict]]:
        """Get all entities from all dictionaries"""
        return self.dictionaries
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict]:
        """Get all entities of a specific type"""
        return self.dictionaries.get(entity_type, [])
    
    def get_entity_details(self, entity_name: str, entity_type: str = None) -> Optional[Dict]:
        """Get detailed information about a specific entity"""
        if entity_type:
            # Search in specific dictionary type
            entity_map = self.entity_maps.get(entity_type, {})
            entity_info = entity_map.get(entity_name.lower())
            if entity_info:
                return entity_info['data']
        else:
            # Search across all dictionaries
            entity_info = self.all_entities.get(entity_name.lower())
            if entity_info:
                return entity_info['data']
        
        return None