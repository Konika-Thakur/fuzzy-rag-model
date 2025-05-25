#!/usr/bin/env python3
"""
LLM Response Generation module for the Enhanced Fuzzy RAG Product Feature Assistant
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional

from config import Config

logger = logging.getLogger(__name__)

# Optional imports for different LLM providers
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests library not available. Install with: pip install requests")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic SDK not available. Install with: pip install anthropic")

class LLMResponseGenerator:
    """Generates comprehensive responses using a Language Model"""
    
    def __init__(self, provider: str = None, model: str = None, api_key: str = None):
        """Initialize the LLM Response Generator"""
        self.provider = provider or Config.LLM_PROVIDER
        self.model = model or Config.LLM_MODEL
        self.temperature = Config.LLM_TEMPERATURE
        self.max_tokens = Config.LLM_MAX_TOKENS
        
        # Initialize the client based on the provider
        self.client = None
        
        if self.provider == "openai" and OPENAI_AVAILABLE:
            openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.client = openai
            logger.info(f"Initialized OpenAI client for model: {self.model}")
        
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
            logger.info(f"Initialized Anthropic client for model: {self.model}")
        
        elif self.provider == "ollama" and REQUESTS_AVAILABLE:
            self.client = "ollama"
            logger.info(f"Using Ollama for model: {self.model}")
        
        else:
            logger.info(f"Using mock LLM provider for {self.model}")
            self.provider = "mock"
    
    def generate_multi_dictionary_response(
        self,
        original_query: str,
        corrected_query: str,
        identified_entities: Dict[str, List[Dict]],
        entity_details: Dict[str, Dict[str, Dict]],
        similar_entities: Dict[str, Dict[str, List[Dict]]]
    ) -> str:
        """Generate a comprehensive response using an LLM for multi-dictionary context"""
        # Construct the prompt
        prompt = self._construct_multi_dictionary_prompt(
            original_query,
            corrected_query,
            identified_entities,
            entity_details,
            similar_entities
        )
        
        logger.info("Generating multi-dictionary LLM response...")
        
        # Call the appropriate LLM API based on provider
        if self.provider == "openai" and self.client is not None:
            return self._call_openai(prompt)
        
        elif self.provider == "anthropic" and self.client is not None:
            return self._call_anthropic(prompt)
        
        elif self.provider == "ollama" and self.client is not None:
            return self._call_ollama(prompt)
        
        else:
            logger.info("Using mock multi-dictionary LLM response")
            return self._generate_mock_multi_dictionary_response(identified_entities)
    
    def generate_response(
        self,
        original_query: str,
        corrected_query: str,
        identified_features: List[Dict],
        feature_details: Dict[str, Dict],
        similar_features: Dict[str, List[Dict]]
    ) -> str:
        """Generate a comprehensive response using an LLM"""
        # Construct the prompt
        prompt = self._construct_prompt(
            original_query,
            corrected_query,
            identified_features,
            feature_details,
            similar_features
        )
        
        logger.info("Generating LLM response...")
        
        # Call the appropriate LLM API based on provider
        if self.provider == "openai" and self.client is not None:
            return self._call_openai(prompt)
        
        elif self.provider == "anthropic" and self.client is not None:
            return self._call_anthropic(prompt)
        
        elif self.provider == "ollama" and self.client is not None:
            return self._call_ollama(prompt)
        
        else:
            logger.info("Using mock LLM response")
            return self._generate_mock_response()
    
    def _construct_multi_dictionary_prompt(
        self,
        original_query: str,
        corrected_query: str,
        identified_entities: Dict[str, List[Dict]],
        entity_details: Dict[str, Dict[str, Dict]],
        similar_entities: Dict[str, Dict[str, List[Dict]]]
    ) -> str:
        """Construct a prompt for the LLM with multi-dictionary context"""
        
        # Count total entities
        total_entities = sum(len(entities) for entities in identified_entities.values())
        
        # Format identified entities by type
        entities_text = self._format_entities_by_type(identified_entities, entity_details)
        
        # Process similar entities
        similar_text = self._format_similar_entities_multi(similar_entities)
        
        # Construct the comprehensive prompt
        prompt = f"""You are a helpful product assistant that understands furniture, styles, places, and features.
The customer has searched for: "{original_query}"

I understood they are looking for: "{corrected_query}"

I identified {total_entities} entities across different categories:

{entities_text}

Here are some related items they might also be interested in:
{similar_text}

Please generate a helpful, comprehensive response to the customer that:
1. Acknowledges their search query
2. Explains each identified entity in detail, organized by category (features, styles, products, places)
3. Shows how these entities work together (e.g., how a style complements certain features)
4. Suggests related items from the similar entities
5. Uses a friendly, informative tone that helps them visualize their ideal setup

Your response should help them understand not just individual items, but how they can create a cohesive look or functional space."""
        
        return prompt.strip()
    
    def _construct_prompt(
        self,
        original_query: str,
        corrected_query: str,
        identified_features: List[Dict],
        feature_details: Dict[str, Dict],
        similar_features: Dict[str, List[Dict]]
    ) -> str:
        """Construct a prompt for the LLM"""
        # Format identified features
        feature_names = [f["name"] for f in identified_features]
        feature_list = ", ".join(feature_names)
        
        # Process feature details into text
        details_text = self._process_feature_details(identified_features, feature_details)
        
        # Process similar features into text
        similar_text = self._process_similar_features(similar_features)
        
        # Construct the prompt
        prompt = f"""You are a helpful product feature assistant that explains product features to customers.
The customer has searched for: "{original_query}"

I understood they are looking for: "{corrected_query}"

The identified product features are: {feature_list}

Here are the details about each identified feature:
{details_text}

Here are some similar or related features they might also be interested in:
{similar_text}

Please generate a helpful, comprehensive response to the customer that:
1. Acknowledges their search query
2. Explains each identified feature in detail, including its benefits
3. Suggests the related features they might also be interested in
4. Uses a friendly, informative tone

Your response should provide valuable information about the features they're looking for and help them make an informed decision."""
        
        return prompt.strip()
    
    def _format_entities_by_type(
        self, 
        identified_entities: Dict[str, List[Dict]], 
        entity_details: Dict[str, Dict[str, Dict]]
    ) -> str:
        """Format entities organized by type for the prompt"""
        formatted = []
        
        type_headers = {
            'features': 'FEATURES',
            'styles': 'STYLES', 
            'products': 'PRODUCTS',
            'places': 'PLACES'
        }
        
        for entity_type, entities in identified_entities.items():
            if not entities:
                continue
                
            type_text = f"{type_headers.get(entity_type, entity_type.upper())}:\n"
            
            for entity in entities:
                entity_name = entity['name']
                details = entity_details.get(entity_type, {}).get(entity_name, {})
                
                entity_text = f"- {entity_name}:\n"
                
                if 'description' in details:
                    entity_text += f"  Description: {details['description']}\n"
                
                # Add type-specific information
                if entity_type == 'features':
                    if 'benefits' in details:
                        benefits = details['benefits']
                        if benefits:
                            entity_text += f"  Benefits: {', '.join(benefits)}\n"
                
                elif entity_type == 'styles':
                    if 'characteristics' in details:
                        characteristics = details['characteristics']
                        if characteristics:
                            entity_text += f"  Characteristics: {', '.join(characteristics)}\n"
                
                elif entity_type == 'products':
                    if 'subcategories' in details:
                        subcategories = details['subcategories']
                        if subcategories:
                            entity_text += f"  Types: {', '.join(subcategories)}\n"
                
                elif entity_type == 'places':
                    if 'typical_furniture' in details:
                        typical_furniture = details['typical_furniture']
                        if typical_furniture:
                            entity_text += f"  Typical furniture: {', '.join(typical_furniture)}\n"
                
                type_text += entity_text
            
            formatted.append(type_text)
        
        return "\n".join(formatted)
    
    def _format_similar_entities_multi(self, similar_entities: Dict[str, Dict[str, List[Dict]]]) -> str:
        """Format similar entities for multi-dictionary context"""
        formatted = []
        
        for entity_type, entity_similarities in similar_entities.items():
            if not entity_similarities:
                continue
                
            for entity_name, similar_list in entity_similarities.items():
                if not similar_list:
                    continue
                    
                similar_text = f"- Similar to {entity_name} ({entity_type}):\n"
                
                for similar in similar_list[:Config.MAX_SIMILAR_FEATURES]:
                    similar_type = similar.get('type', 'unknown')
                    similar_text += f"  - {similar['name']} ({similar_type})"
                    
                    if 'description' in similar:
                        similar_text += f": {similar['description']}"
                    
                    if 'similarity' in similar:
                        similar_text += f" (similarity: {similar['similarity']:.2f})"
                    
                    similar_text += "\n"
                
                formatted.append(similar_text)
        
        return "\n".join(formatted)
    
    def _process_feature_details(self, features: List[Dict], details: Dict[str, Dict]) -> str:
        """Process feature details into a structured text format for the prompt"""
        formatted = []
        
        for feature in features:
            name = feature['name']
            feature_data = details.get(name, {})
            
            feature_text = f"- {name}:\n"
            
            if 'description' in feature_data:
                feature_text += f"  Description: {feature_data['description']}\n"
                
            if 'benefits' in feature_data:
                benefits = feature_data['benefits']
                if benefits:
                    feature_text += "  Benefits:\n"
                    for benefit in benefits:
                        feature_text += f"    - {benefit}\n"
            
            if 'technical_specs' in feature_data:
                specs = feature_data['technical_specs']
                if specs:
                    feature_text += "  Technical Specifications:\n"
                    for key, value in specs.items():
                        feature_text += f"    - {key}: {value}\n"
            
            formatted.append(feature_text)
        
        return "\n".join(formatted)
    
    def _process_similar_features(self, similar_features: Dict[str, List[Dict]]) -> str:
        """Process similar features into a structured text format for the prompt"""
        formatted = []
        
        for feature_name, similar_list in similar_features.items():
            feature_text = f"- Similar to {feature_name}:\n"
            
            for similar in similar_list[:Config.MAX_SIMILAR_FEATURES]:
                feature_text += f"  - {similar['name']}"
                
                if 'description' in similar:
                    feature_text += f": {similar['description']}"
                    
                if 'similarity' in similar:
                    feature_text += f" (similarity: {similar['similarity']:.2f})"
                
                feature_text += "\n"
            
            formatted.append(feature_text)
        
        return "\n".join(formatted)
    
    def _generate_mock_multi_dictionary_response(self, identified_entities: Dict[str, List[Dict]]) -> str:
        """Generate a mock response for multi-dictionary context when LLM API is unavailable"""
        entity_types_found = [entity_type for entity_type, entities in identified_entities.items() if entities]
        total_entities = sum(len(entities) for entities in identified_entities.values())
        
        if not entity_types_found:
            return """I understand you're looking for specific items, but I wasn't able to identify the exact features, styles, products, or places you mentioned. 

Could you please provide more details about what you're looking for? For example:
- What type of furniture (chair, desk, sofa)?
- What style you prefer (modern, industrial, scandinavian)?
- What features are important (lumbar support, adjustable height)?
- Where you'll use it (office, living room, bedroom)?

This will help me provide better recommendations for your needs."""
        
        response_parts = [
            f"I understand you're looking for items across {len(entity_types_found)} categories. Let me break down what I found:"
        ]
        
        # Add category-specific responses
        for entity_type, entities in identified_entities.items():
            if not entities:
                continue
                
            entity_names = [entity['name'] for entity in entities]
            
            if entity_type == 'features':
                response_parts.append(f"\n**Features**: {', '.join(entity_names)}")
                response_parts.append("These features will enhance functionality and user experience, providing specific benefits for your intended use.")
            
            elif entity_type == 'styles':
                response_parts.append(f"\n**Styles**: {', '.join(entity_names)}")
                response_parts.append("These design styles will create a cohesive aesthetic that matches your preferences and space.")
            
            elif entity_type == 'products':
                response_parts.append(f"\n**Products**: {', '.join(entity_names)}")
                response_parts.append("These furniture pieces will serve as the foundation for your space and daily activities.")
            
            elif entity_type == 'places':
                response_parts.append(f"\n**Places**: {', '.join(entity_names)}")
                response_parts.append("Understanding the intended space helps ensure the items will work well in that environment.")
        
        response_parts.append(f"\nThe combination of these {total_entities} elements creates a comprehensive picture of what you're looking for. Each category complements the others to create a functional and aesthetically pleasing result.")
        
        response_parts.append("\nWould you like me to suggest specific products that combine these elements, or do you need more information about any particular category?")
        
        return "\n".join(response_parts)
    
    def _generate_mock_response(self) -> str:
        """Generate a mock response when LLM API is unavailable"""
        return """I understand you're looking for specific product features.

Based on your search, I can see you're interested in high-quality features that enhance functionality and user experience. These features are designed to provide excellent value and meet your specific needs.

Each feature offers unique benefits that contribute to the overall quality and performance of the product. The combination of these features creates a comprehensive solution that addresses multiple user requirements.

You might also be interested in exploring additional features that complement your selection and provide even more value for your specific use case."""
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API to generate a response"""
        try:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful product feature assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return self._generate_mock_response()
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API to generate a response"""
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=f"Human: {prompt}\n\nAssistant:",
                temperature=self.temperature,
                max_tokens_to_sample=self.max_tokens
            )
            
            return response.completion
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return self._generate_mock_response()
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API to generate a response"""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available for Ollama API")
            return self._generate_mock_response()
            
        try:
            response = requests.post(
                f"http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Error from Ollama API: {response.status_code} {response.text}")
                return self._generate_mock_response()
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return self._generate_mock_response()
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported LLM providers"""
        providers = ["mock"]  # Always available
        
        if OPENAI_AVAILABLE:
            providers.append("openai")
        if ANTHROPIC_AVAILABLE:
            providers.append("anthropic")
        if REQUESTS_AVAILABLE:
            providers.append("ollama")
            
        return providers
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get current provider status and configuration"""
        return {
            "current_provider": self.provider,
            "current_model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "client_initialized": self.client is not None,
            "supported_providers": self.get_supported_providers(),
            "library_availability": {
                "openai": OPENAI_AVAILABLE,
                "anthropic": ANTHROPIC_AVAILABLE,
                "requests": REQUESTS_AVAILABLE
            }
        }
    
    def update_configuration(self, provider: str = None, model: str = None, 
                           temperature: float = None, max_tokens: int = None) -> bool:
        """Update LLM configuration dynamically"""
        try:
            if provider and provider != self.provider:
                self.provider = provider
                # Re-initialize client for new provider
                self.__init__(provider, model or self.model)
                
            if model:
                self.model = model
                
            if temperature is not None:
                self.temperature = temperature
                
            if max_tokens is not None:
                self.max_tokens = max_tokens
                
            logger.info(f"Updated LLM configuration: {self.provider}/{self.model}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating LLM configuration: {e}")
            return False