#!/usr/bin/env python3
"""
Custom configuration for the Enhanced Fuzzy RAG Product Feature Assistant
- Uses Llama 3.2 (3B) via Ollama
- Qdrant for vector storage
- BAAI/bge-small-en for embeddings
"""

import os
import logging
from Enhanced_Fuzzy_RAG_Product_Feature_Assistant import EnhancedProductFeatureAssistant
from DictionaryConverter import DictionaryConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_custom_assistant(
    dictionaries_path: str = "dictionaries",
    use_sample_data: bool = True,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "product_features"
) -> EnhancedProductFeatureAssistant:
    """
    Set up the assistant with custom configuration
    
    Args:
        dictionaries_path: Path to dictionary files
        use_sample_data: Whether to use sample data
        qdrant_url: URL for Qdrant
        collection_name: Collection name in Qdrant
        
    Returns:
        Configured assistant
    """
    # Create the assistant with custom settings
    assistant = EnhancedProductFeatureAssistant(
        dictionaries_path=dictionaries_path,
        embedding_model="BAAI/bge-small-en",
        device="cpu",  # Change to "cuda" if you have a GPU
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        llm_model="llama3:3b",
        llm_provider="ollama"
    )
    
    # Load sample data if needed
    if use_sample_data and not os.path.exists(os.path.join(dictionaries_path, "featuresDictionary.json")):
        logger.info("Loading sample dictionary data...")
        DictionaryConverter.generate_sample_dictionaries(dictionaries_path)
    
    # Initialize system
    logger.info("Initializing assistant...")
    assistant.initialize_from_dictionaries()
    
    return assistant

def main():
    """Run the assistant with custom configuration"""
    # Ensure dictionaries directory exists
    os.makedirs("dictionaries", exist_ok=True)
    
    # Set up the custom assistant
    assistant = setup_custom_assistant(use_sample_data=True)
    
    # Test query
    test_query = "Looking for a chair with highbak and metl legs"
    logger.info(f"Processing test query: {test_query}")
    
    # Process the query
    response = assistant.process_query(test_query)
    
    # Print the response
    print("\n" + "="*60)
    print("PRODUCT FEATURE ASSISTANT RESPONSE")
    print("="*60)
    
    print(f"\nOriginal Query: {response['original_query']}")
    print(f"Interpreted Query: {response['corrected_query']}")
    
    print("\nIdentified Features:")
    for feature in response['identified_features']:
        print(f"- {feature['name']} (confidence: {feature['confidence']:.2f})")
    
    print("\nFeature Explanation:")
    print(response['explanation'])
    
    print("\nSimilar Features You Might Like:")
    for feature_name, similar_list in response['similar_features'].items():
        for similar in similar_list:
            print(f"- {similar['name']} (similar to {feature_name})")
    
    print("\n" + "="*60)
    
    # Interactive mode
    print("\nEnhanced Fuzzy RAG Product Feature Assistant")
    print("Type 'exit' or 'quit' to end the session")
    
    while True:
        query = input("\nEnter your query: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not query.strip():
            continue
            
        try:
            response = assistant.process_query(query)
            
            print("\n" + "="*60)
            print("PRODUCT FEATURE ASSISTANT RESPONSE")
            print("="*60)
            
            print(f"\nOriginal Query: {response['original_query']}")
            print(f"Interpreted Query: {response['corrected_query']}")
            
            print("\nIdentified Features:")
            for feature in response['identified_features']:
                print(f"- {feature['name']} (confidence: {feature['confidence']:.2f})")
            
            print("\nFeature Explanation:")
            print(response['explanation'])
            
            print("\nSimilar Features You Might Like:")
            for feature_name, similar_list in response['similar_features'].items():
                for similar in similar_list:
                    print(f"- {similar['name']} (similar to {feature_name})")
            
            print("\n" + "="*60)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()