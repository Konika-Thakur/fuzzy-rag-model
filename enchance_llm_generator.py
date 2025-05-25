#!/usr/bin/env python3
"""
Enhanced LLM Response Generation with Unknown Token Classification
Identifies brands, materials, colors, designers, etc. for unmatched tokens
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Optional imports for different LLM providers
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - Ollama support disabled")

class EnhancedLLMGenerator:
    """Enhanced LLM generator with token classification capabilities"""
    
    def __init__(self):
        """Initialize the LLM generator"""
        self.provider = "ollama"
        self.model = "llama3.2:3b"
        self.temperature = 0.7
        self.max_tokens = 512
    
    def classify_unknown_tokens(self, tokens: List[str]) -> Dict[str, Dict[str, Any]]:
        """Classify unknown tokens using LLM or rule-based fallback"""
        if not tokens:
            return {}
        
        # Use rule-based classification (works without LLM)
        return self._rule_based_classification(tokens)
    
    def _rule_based_classification(self, tokens: List[str]) -> Dict[str, Dict[str, Any]]:
        """Enhanced rule-based classification for common cases"""
        classifications = {}
        
        for token in tokens:
            token_lower = token.lower()
            
            # Brand classification
            brands = ['ikea', 'herman', 'miller', 'steelcase', 'knoll', 'west', 'elm', 'cb2', 'crate', 'barrel', 'pottery', 'barn', 'wayfair', 'amazon', 'target', 'walmart']
            if any(brand in token_lower for brand in brands):
                classifications[token] = {
                    "type": "BRAND",
                    "confidence": 0.9,
                    "explanation": f"Recognized furniture brand: {token}"
                }
            
            # Designer classification
            elif any(designer in token_lower for designer in ['eames', 'jacobsen', 'starck', 'aalto', 'wegner', 'saarinen', 'nelson', 'bertoia']):
                classifications[token] = {
                    "type": "DESIGNER",
                    "confidence": 0.9,
                    "explanation": f"Recognized furniture designer: {token}"
                }
            
            # Material classification
            elif any(material in token_lower for material in ['wood', 'metal', 'leather', 'fabric', 'plastic', 'steel', 'oak', 'pine', 'maple', 'walnut', 'mahogany', 'aluminum', 'chrome', 'brass', 'glass']):
                classifications[token] = {
                    "type": "MATERIAL", 
                    "confidence": 0.8,
                    "explanation": f"Common furniture material: {token}"
                }
            
            # Color classification
            elif any(color in token_lower for color in ['black', 'white', 'red', 'blue', 'green', 'brown', 'gray', 'grey', 'beige', 'tan', 'navy', 'cream', 'ivory', 'charcoal']):
                classifications[token] = {
                    "type": "COLOR",
                    "confidence": 0.8,
                    "explanation": f"Color descriptor: {token}"
                }
            
            # Size classification
            elif any(size in token_lower for size in ['large', 'small', 'big', 'compact', 'oversized', 'mini', 'xl', 'xs', 'medium', 'king', 'queen', 'twin', 'full']):
                classifications[token] = {
                    "type": "SIZE",
                    "confidence": 0.7,
                    "explanation": f"Size descriptor: {token}"
                }
            
            # Dimension classification
            elif any(dim in token for dim in ['inch', 'cm', 'mm', 'foot', 'ft', '"', "'"]) or token.replace('.', '').replace('-', '').isdigit():
                classifications[token] = {
                    "type": "DIMENSION",
                    "confidence": 0.8,
                    "explanation": f"Dimension measurement: {token}"
                }
            
            else:
                classifications[token] = {
                    "type": "OTHER",
                    "confidence": 0.5,
                    "explanation": f"Unknown furniture-related term: {token}"
                }
        
        return classifications
    
    def generate_comprehensive_response(self, 
                                      matched_entities: Dict[str, List[Dict]], 
                                      classified_tokens: Dict[str, Dict[str, Any]],
                                      original_query: str) -> str:
        """Generate a comprehensive response combining matched entities and classified tokens"""
        
        response_parts = []
        
        # Count total entities
        total_entities = sum(len(entities) for entities in matched_entities.values())
        
        if total_entities == 0 and not classified_tokens:
            return "I couldn't identify specific furniture-related terms in your query. Please try rephrasing with more specific details about furniture, styles, features, or places."
        
        response_parts.append(f"Based on your query '{original_query}', I found:")
        
        # Add matched entities by type
        for entity_type, entities in matched_entities.items():
            if entities:
                entity_names = [e["name"] for e in entities[:3]]
                response_parts.append(f"\n**{entity_type.title()}**: {', '.join(entity_names)}")
        
        # Add classified tokens
        if classified_tokens:
            classified_parts = []
            for token, classification in classified_tokens.items():
                classified_parts.append(f"{token} ({classification['type'].lower()})")
            response_parts.append(f"\n**Additional terms identified**: {', '.join(classified_parts)}")
        
        # Add contextual advice
        if matched_entities.get('products') and matched_entities.get('places'):
            product = matched_entities['products'][0]['name']
            place = matched_entities['places'][0]['name']
            response_parts.append(f"\nFor a {product.lower()} in your {place.lower()}, consider factors like size, comfort, and functionality that match your space and needs.")
        
        return "\n".join(response_parts)
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get the status of the current LLM provider"""
        return {
            "provider": "rule-based",
            "model": "rule-based classification",
            "available": True,
            "error": None
        }