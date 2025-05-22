#!/usr/bin/env python3
"""
LLM Response Generation module for the Enhanced Fuzzy RAG Product Feature Assistant
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional

# Optional imports for different LLM providers
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests library not available. Install with: pip install requests")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI SDK not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic SDK not available. Install with: pip install anthropic")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMResponseGenerator:
    """
    Generates comprehensive responses using a Language Model
    """
    
    def __init__(
        self,
        model: str = "llama3:3b",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        provider: str = "ollama"
    ):
        """
        Initialize the LLM Response Generator
        
        Args:
            model: Name/identifier of the LLM to use
            api_key: API key for the LLM provider
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            provider: LLM provider ("openai", "anthropic", "ollama", "mock")
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        
        # Initialize the client based on the provider
        self.client = None
        
        if provider == "openai" and OPENAI_AVAILABLE:
            openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.client = openai
            logger.info(f"Initialized OpenAI client for model: {model}")
        
        elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
            logger.info(f"Initialized Anthropic client for model: {model}")
        
        elif provider == "ollama" and REQUESTS_AVAILABLE:
            # For local Ollama models, just set flag as we'll use requests directly
            self.client = "ollama"
            logger.info(f"Using Ollama for model: {model}")
        
        else:
            logger.warning(f"Using mock LLM provider for {model}")
            self.provider = "mock"
    
    def _process_feature_details(self, features: List[Dict], details: Dict[str, Dict]) -> str:
        """
        Process feature details into a structured text format for the prompt
        
        Args:
            features: List of identified features
            details: Dictionary of feature details keyed by feature name
            
        Returns:
            Formatted feature details as text
        """
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
        """
        Process similar features into a structured text format for the prompt
        
        Args:
            similar_features: Dictionary of similar features keyed by original feature name
            
        Returns:
            Formatted similar features as text
        """
        formatted = []
        
        for feature_name, similar_list in similar_features.items():
            feature_text = f"- Similar to {feature_name}:\n"
            
            for similar in similar_list[:3]:  # Limit to top 3 similar features
                feature_text += f"  - {similar['name']}"
                
                if 'description' in similar:
                    feature_text += f": {similar['description']}"
                    
                if 'similarity' in similar:
                    feature_text += f" (similarity: {similar['similarity']:.2f})"
                
                feature_text += "\n"
            
            formatted.append(feature_text)
        
        return "\n".join(formatted)
    
    def _construct_prompt(
        self,
        original_query: str,
        corrected_query: str,
        identified_features: List[Dict],
        feature_details: Dict[str, Dict],
        similar_features: Dict[str, List[Dict]]
    ) -> str:
        """
        Construct a prompt for the LLM
        
        Args:
            original_query: Original user query
            corrected_query: Corrected query with canonical feature names
            identified_features: List of identified features
            feature_details: Dictionary of feature details
            similar_features: Dictionary of similar features
            
        Returns:
            Constructed prompt string
        """
        # Format identified features
        feature_names = [f["name"] for f in identified_features]
        feature_list = ", ".join(feature_names)
        
        # Process feature details into text
        details_text = self._process_feature_details(identified_features, feature_details)
        
        # Process similar features into text
        similar_text = self._process_similar_features(similar_features)
        
        # Construct the prompt
        prompt = f"""
You are a helpful product feature assistant that explains product features to customers.
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

Your response should provide valuable information about the features they're looking for and help them make an informed decision.
        """
        
        return prompt.strip()
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI API to generate a response
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Generated response text
        """
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
        """
        Call Anthropic API to generate a response
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Generated response text
        """
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
        """
        Call Ollama API to generate a response
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Generated response text
        """
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available for Ollama API")
            return self._generate_mock_response()
            
        try:
            # Assuming Ollama API is running locally
            response = requests.post(
                f"http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_length": self.max_tokens
                },
                timeout=60
            )
            
            if response.status_code == 200:
                # Ollama returns streaming responses, collect all text
                full_response = ""
                json_data = response.json()
                if isinstance(json_data, dict) and "response" in json_data:
                    full_response = json_data["response"]
                else:
                    # Parse streaming response if needed
                    full_response = response.text
                
                return full_response.strip()
            else:
                logger.error(f"Error from Ollama API: {response.status_code} {response.text}")
                return self._generate_mock_response()
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return self._generate_mock_response()
    
    def _generate_mock_response(self) -> str:
        """
        Generate a mock response when LLM API is unavailable
        
        Returns:
            Mock response text
        """
        return """
I understand you're looking for a chair with a HighBack and Metal Legs.

A chair with a HighBack provides excellent ergonomic support for your upper back, neck, and head, which is ideal for longer sitting sessions. This design helps maintain proper posture and reduces strain during extended use.

The Metal Legs offer several advantages: they provide excellent stability while maintaining a sleek, minimalist appearance. Typically made from steel, aluminum, or other alloys, these legs are stronger than wood alternatives and can support greater weight while appearing visually lighter.

You might also be interested in chairs with:
- Lumbar Support: This feature provides additional support for your lower back, further enhancing comfort and ergonomics.
- Industrial Design: This aesthetic complements metal legs perfectly and creates a contemporary look for modern spaces.

Please let me know if you have any other questions about these features or if you'd like recommendations for specific chairs that include these features.
        """
    
    def generate_response(
        self,
        original_query: str,
        corrected_query: str,
        identified_features: List[Dict],
        feature_details: Dict[str, Dict],
        similar_features: Dict[str, List[Dict]]
    ) -> str:
        """
        Generate a comprehensive response using an LLM
        
        Args:
            original_query: Original user query
            corrected_query: Corrected query with canonical feature names
            identified_features: List of identified features
            feature_details: Dictionary of feature details
            similar_features: Dictionary of similar features
            
        Returns:
            Generated response text
        """
        # Construct the prompt
        prompt = self._construct_prompt(
            original_query,
            corrected_query,
            identified_features,
            feature_details,
            similar_features
        )
        
        logger.info("Constructed LLM prompt, generating response...")
        
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


if __name__ == "__main__":
    # Test code for the LLM Response Generator
    generator = LLMResponseGenerator(provider="mock")
    
    # Example data
    original_query = "Looking for a chair with highbak and metl legs"
    corrected_query = "Looking for a chair with HighBack and Metal Legs"
    identified_features = [
        {"name": "HighBack", "confidence": 0.82},
        {"name": "Metal Legs", "confidence": 0.86}
    ]
    feature_details = {
        "HighBack": {
            "description": "Chair back that extends above shoulder height",
            "benefits": ["Better posture", "Reduced neck strain", "Head support"]
        },
        "Metal Legs": {
            "description": "Chair legs made of metal material",
            "benefits": ["Durability", "Modern look", "Stability"]
        }
    }
    similar_features = {
        "HighBack": [
            {"name": "Lumbar Support", "description": "Support for the lower back", "similarity": 0.85},
            {"name": "Ergonomic Design", "description": "Designed for body comfort", "similarity": 0.82}
        ],
        "Metal Legs": [
            {"name": "Industrial Design", "description": "Modern industrial aesthetic", "similarity": 0.88},
            {"name": "Aluminum Frame", "description": "Lightweight but strong", "similarity": 0.84}
        ]
    }
    
    # Generate response
    response = generator.generate_response(
        original_query,
        corrected_query,
        identified_features,
        feature_details,
        similar_features
    )
    
    print("Generated LLM Response:")
    print(response)