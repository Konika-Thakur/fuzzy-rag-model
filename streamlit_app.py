#!/usr/bin/env python3
"""
Standalone Streamlit interface for the Enhanced Fuzzy RAG Product Feature Assistant.
This version implements a simplified version of the assistant directly within the file.
"""

import os
import json
import re
import logging
import difflib
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Enhanced Fuzzy RAG Product Feature Assistant",
    page_icon="🔍",
    layout="wide"
)

###########################################
# Simplified implementation of key classes
###########################################

class SimpleFuzzyMatcher:
    """Simplified fuzzy matcher for product features"""
    
    def __init__(self, dictionaries_path: str):
        self.dictionaries_path = dictionaries_path
        self.features_full = []
        self.features_map = {}
        
        # Load feature dictionary
        self._load_features_dictionary()
    
    def _load_features_dictionary(self):
        """Load features dictionary from a JSON file"""
        filepath = os.path.join(self.dictionaries_path, "featuresDictionary.json")
        
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.features_full = json.load(f)
                
            # Create mapping from all variations to canonical names
            for feature in self.features_full:
                canonical_name = feature.get("feature", "")
                if not canonical_name:
                    continue
                    
                # Add canonical name to mapping (lowercase for case-insensitive matching)
                self.features_map[canonical_name.lower()] = canonical_name
                
                # Add spelling variations
                for variation in feature.get("spelling_variations", []):
                    self.features_map[variation.lower()] = canonical_name
                    
                # Add synonyms
                for synonym in feature.get("synonyms", []):
                    self.features_map[synonym.lower()] = canonical_name
        except Exception as e:
            st.error(f"Error loading feature dictionary: {e}")
    
    def extract_features(self, query: str) -> List[Dict]:
        """Extract features from a query using fuzzy matching"""
        identified_features = []
        
        # Simple check for known features and variations
        for term, canonical in self.features_map.items():
            if term.lower() in query.lower():
                # Check for multi-word terms
                if " " in term:
                    identified_features.append({
                        'name': canonical,
                        'original_text': term,
                        'confidence': 0.95
                    })
                    continue
                
                # Check for whole word match for single word terms
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, query, re.IGNORECASE):
                    identified_features.append({
                        'name': canonical,
                        'original_text': term,
                        'confidence': 0.9
                    })
                    continue
        
        # Try fuzzy matching for remaining terms
        words = query.split()
        for word in words:
            if len(word) < 3:
                continue
                
            # Check if this word is similar to any feature term
            matches = difflib.get_close_matches(word.lower(), self.features_map.keys(), n=1, cutoff=0.7)
            if matches:
                match = matches[0]
                canonical = self.features_map[match]
                
                # Check if we already found this feature
                if any(f['name'] == canonical for f in identified_features):
                    continue
                    
                identified_features.append({
                    'name': canonical,
                    'original_text': word,
                    'confidence': 0.7
                })
        
        return identified_features
    
    def correct_query(self, query: str, features: List[Dict]) -> str:
        """Create a corrected version of the query with canonical feature names"""
        corrected = query
        
        # Simple replacement
        for feature in features:
            if 'original_text' in feature and 'name' in feature:
                corrected = re.sub(
                    r'\b' + re.escape(feature['original_text']) + r'\b',
                    feature['name'],
                    corrected,
                    flags=re.IGNORECASE
                )
        
        return corrected
    
    def get_all_features(self) -> List[Dict]:
        """Get all features from the dictionary"""
        return self.features_full


class SimpleVectorRetriever:
    """Simplified vector retriever that returns mock data"""
    
    def __init__(self, features_data: List[Dict] = None):
        self.features_data = features_data or []
        self.feature_details = {}
        
        # Process feature data
        for feature in self.features_data:
            name = feature.get('feature', '')
            if not name:
                continue
                
            # Store feature details
            self.feature_details[name] = {
                'feature_name': name,
                'description': feature.get('description', ''),
                'benefits': feature.get('benefits', [])
            }
    
    def get_feature_details(self, feature_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific feature"""
        if feature_name in self.feature_details:
            return self.feature_details[feature_name]
            
        # Return default data if not found
        return {
            "feature_name": feature_name,
            "description": f"Description for {feature_name}",
            "benefits": [f"Benefit 1 of {feature_name}", f"Benefit 2 of {feature_name}"]
        }
    
    def get_similar_features(self, feature_name: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Get features similar to a given feature"""
        # Define some default similar features for common cases
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
            
        # For other features, return generic similar features
        return [
            {'name': 'Ergonomic Design', 'description': 'Designed for comfort and posture', 'similarity': 0.75},
            {'name': 'Premium Materials', 'description': 'High-quality construction materials', 'similarity': 0.70}
        ]


class SimpleMockLLM:
    """Simplified mock LLM that generates responses about product features"""
    
    def generate_response(
        self,
        original_query: str,
        corrected_query: str,
        identified_features: List[Dict],
        feature_details: Dict[str, Dict],
        similar_features: Dict[str, List[Dict]]
    ) -> str:
        """Generate a response based on the identified features"""
        response = [f"I understand you're looking for {' and '.join([f['name'] for f in identified_features])}."]
        
        # Add descriptions for each feature
        for feature in identified_features:
            name = feature['name']
            details = feature_details.get(name, {})
            description = details.get('description', f"A {name} is a valuable feature")
            benefits = details.get('benefits', [])
            
            response.append(f"\n{name}: {description}")
            
            if benefits:
                response.append("Benefits:")
                for benefit in benefits:
                    response.append(f"- {benefit}")
        
        # Add similar feature suggestions
        if similar_features:
            response.append("\nYou might also be interested in:")
            for feature_name, similar_list in similar_features.items():
                for similar in similar_list:
                    response.append(f"- {similar['name']} (similar to {feature_name}): {similar.get('description', '')}")
        
        return "\n".join(response)


class SimpleProductAssistant:
    """Simplified product feature assistant that integrates the components"""
    
    def __init__(self, dictionaries_path: str = "dictionaries"):
        self.dictionaries_path = dictionaries_path
        self.fuzzy_matcher = SimpleFuzzyMatcher(dictionaries_path)
        self.vector_retriever = None  # Will be initialized later
        self.llm = SimpleMockLLM()
        
    def initialize_from_dictionaries(self):
        """Initialize the system from the dictionaries"""
        features_data = self.fuzzy_matcher.get_all_features()
        self.vector_retriever = SimpleVectorRetriever(features_data)
        return True
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query"""
        # Extract features using fuzzy matching
        identified_features = self.fuzzy_matcher.extract_features(query)
        
        # Correct the query
        corrected_query = self.fuzzy_matcher.correct_query(query, identified_features)
        
        # Get feature details and similar features
        feature_details = {}
        similar_features = {}
        
        for feature in identified_features:
            # Get details for each identified feature
            details = self.vector_retriever.get_feature_details(feature['name'])
            feature_details[feature['name']] = details
            
            # Get similar features
            similar = self.vector_retriever.get_similar_features(feature['name'])
            similar_features[feature['name']] = similar
        
        # Generate LLM response
        explanation = self.llm.generate_response(
            original_query=query,
            corrected_query=corrected_query,
            identified_features=identified_features,
            feature_details=feature_details,
            similar_features=similar_features
        )
        
        # Create the final response
        response = {
            'original_query': query,
            'corrected_query': corrected_query,
            'identified_features': identified_features,
            'feature_details': feature_details,
            'similar_features': similar_features,
            'explanation': explanation
        }
        
        return response


class SampleDictionaryGenerator:
    """Generates sample dictionaries for testing"""
    
    @staticmethod
    def generate_sample_dictionaries(output_dir: str = "dictionaries") -> bool:
        """Generate sample dictionary files for testing"""
        try:
            # Create the output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Sample features
            features = [
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
                },
                {
                    "feature": "Adjustable Height",
                    "description": "Ability to change the height of the chair",
                    "spelling_variations": ["adjustible height", "height adjustment", "hight adjustable"],
                    "synonyms": ["gas lift", "pneumatic lift", "variable height"],
                    "benefits": ["Customizable comfort", "Ergonomic positioning", "Versatility"]
                },
                {
                    "feature": "Swivel Base",
                    "description": "Base that allows the chair to rotate 360 degrees",
                    "spelling_variations": ["swivel", "rotating base", "swivl base"],
                    "synonyms": ["rotating mechanism", "turntable base", "pivoting base"],
                    "benefits": ["Easy movement", "Access to workspace", "Reduced strain"]
                }
            ]
            
            # Write the features dictionary
            with open(os.path.join(output_dir, "featuresDictionary.json"), 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error generating sample dictionaries: {e}")
            return False


###########################################
# Streamlit app functions
###########################################

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'history' not in st.session_state:
    st.session_state.history = []

def setup_assistant(
    dictionaries_path: str,
    use_sample_data: bool
) -> SimpleProductAssistant:
    """Set up the product feature assistant"""
    # Create the assistant
    assistant = SimpleProductAssistant(dictionaries_path=dictionaries_path)
    
    # Load sample data if needed
    if use_sample_data and not os.path.exists(os.path.join(dictionaries_path, "featuresDictionary.json")):
        logger.info("Loading sample dictionary data...")
        SampleDictionaryGenerator.generate_sample_dictionaries(dictionaries_path)
    
    # Initialize system
    logger.info("Initializing assistant...")
    assistant.initialize_from_dictionaries()
    
    return assistant

def display_response(response: Dict[str, Any]) -> None:
    """Display the response in a nicely formatted way"""
    st.write("### Interpreted Query")
    st.info(f"'{response['corrected_query']}'")
    
    st.write("### Identified Features")
    for feature in response['identified_features']:
        st.write(f"- **{feature['name']}** (confidence: {feature['confidence']:.2f})")
    
    st.write("### Explanation")
    st.markdown(response['explanation'])
    
    st.write("### Similar Features You Might Like")
    for feature_name, similar_list in response['similar_features'].items():
        for similar in similar_list:
            st.write(f"- **{similar['name']}** (similar to {feature_name}): {similar.get('description', '')}")

def display_feature_dictionaries(dictionaries_path: str) -> None:
    """Display the contents of feature dictionaries"""
    features_path = os.path.join(dictionaries_path, "featuresDictionary.json")
    
    if os.path.exists(features_path):
        with open(features_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
            
        # Display features in an expandable section
        with st.expander("View Available Features"):
            for feature in features:
                feature_name = feature.get('feature', '')
                description = feature.get('description', '')
                
                if feature_name:
                    st.markdown(f"**{feature_name}**")
                    st.write(description)
                    
                    # Display variations
                    variations = feature.get('spelling_variations', [])
                    synonyms = feature.get('synonyms', [])
                    
                    if variations or synonyms:
                        all_variations = ', '.join(variations + synonyms)
                        st.caption(f"Variations: {all_variations}")
                    
                    st.divider()

###########################################
# Main Streamlit app
###########################################

def main():
    """Main function for the Streamlit app"""
    st.title("Enhanced Fuzzy RAG Product Feature Assistant")
    st.markdown(
        """
        This system processes natural language queries that may contain typos or non-standard terminology, 
        identifies product features, and provides comprehensive explanations.
        """
    )
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Dictionary settings
        st.subheader("Dictionary Settings")
        dictionaries_path = st.text_input("Dictionaries Path", value="dictionaries")
        use_sample_data = st.checkbox("Use Sample Data", value=True)
        
        # Initialize button
        if st.button("Initialize System"):
            with st.spinner("Initializing system..."):
                try:
                    # Create dictionaries directory if it doesn't exist
                    os.makedirs(dictionaries_path, exist_ok=True)
                    
                    # Set up the assistant
                    st.session_state.assistant = setup_assistant(
                        dictionaries_path=dictionaries_path,
                        use_sample_data=use_sample_data
                    )
                    
                    st.session_state.initialized = True
                    st.success("System initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing system: {e}")
                    logger.error(f"Error initializing system: {e}")
        
        # Display system status
        st.subheader("System Status")
        if st.session_state.initialized:
            st.success("System is initialized and ready!")
        else:
            st.warning("System needs to be initialized")
            
        # Example queries
        st.subheader("Example Queries")
        example_queries = [
            "Looking for a chair with highbak and metl legs",
            "Need a desk with adjustble height",
            "Want a chair with lumbar suport",
            "Modern chair with swivel base",
            "Chair with high back and adjustable height"
        ]
        
        for query in example_queries:
            if st.button(query):
                if st.session_state.initialized:
                    st.session_state.query = query
                    st.experimental_rerun()
                else:
                    st.warning("Please initialize the system first")
    
    # Main content area
    if st.session_state.initialized:
        # Display feature dictionaries
        display_feature_dictionaries(dictionaries_path)
        
        # Query input
        query = st.text_input(
            "Enter your query:",
            value=st.session_state.get("query", ""),
            placeholder="e.g., Looking for a chair with highbak and metl legs",
            key="query_input"
        )
        
        # Process query
        if query and st.button("Process Query"):
            with st.spinner("Processing query..."):
                try:
                    response = st.session_state.assistant.process_query(query)
                    
                    # Add to history
                    st.session_state.history.append({
                        "query": query,
                        "response": response
                    })
                    
                    # Display response
                    st.subheader("Response")
                    display_response(response)
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    logger.error(f"Error processing query: {e}")
        
        # Display history
        if st.session_state.history:
            st.subheader("History")
            
            for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Show only last 5 items
                with st.expander(f"Query: {item['query']}"):
                    display_response(item['response'])
    else:
        st.info("Please initialize the system using the sidebar controls")
        
        # Display sample data info
        st.markdown("### Sample Data")
        st.markdown(
            """
            The system comes with sample product feature data including:
            - **HighBack**: Chair back that extends above shoulder height
            - **Metal Legs**: Legs made of metal material
            - **Lumbar Support**: Additional support for the lower back area
            - **Adjustable Height**: Ability to change the height of the chair
            - **Swivel Base**: Base that allows the chair to rotate 360 degrees
            
            Initialize the system with 'Use Sample Data' checked to try out these features.
            """
        )

if __name__ == "__main__":
    main()