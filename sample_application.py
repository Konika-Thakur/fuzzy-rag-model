#!/usr/bin/env python3
"""
Sample application demonstrating the Enhanced Fuzzy RAG Product Feature Assistant
"""

import os
import logging
import argparse
from typing import Dict, Any

from enhanced_product_assistant import EnhancedProductFeatureAssistant
from dictionary_converter import DictionaryConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_assistant(
    dictionaries_path: str = "dictionaries",
    use_sample_data: bool = False,
    llm_model: str = "llama3",
    llm_provider: str = "ollama"
) -> EnhancedProductFeatureAssistant:
    """
    Set up the Enhanced Product Feature Assistant
    
    Args:
        dictionaries_path: Path to the dictionary files
        use_sample_data: Whether to use sample data
        llm_model: Name of the LLM model to use
        llm_provider: Provider for the LLM
        
    Returns:
        Initialized EnhancedProductFeatureAssistant
    """
    # Create the assistant
    assistant = EnhancedProductFeatureAssistant(
        dictionaries_path=dictionaries_path,
        llm_model=llm_model,
        llm_provider=llm_provider
    )
    
    # Load sample data if needed
    if use_sample_data:
        logger.info("Loading sample dictionary data...")
        DictionaryConverter.generate_sample_dictionaries(dictionaries_path)
    
    # Initialize system
    logger.info("Initializing assistant...")
    assistant.initialize_from_dictionaries()
    
    return assistant

def process_query(assistant: EnhancedProductFeatureAssistant, query: str) -> Dict[str, Any]:
    """
    Process a user query and return the response
    
    Args:
        assistant: The initialized assistant
        query: User query to process
        
    Returns:
        Processed response
    """
    logger.info(f"Processing query: {query}")
    
    # Process the query
    response = assistant.process_query(query)
    
    return response

def print_response(response: Dict[str, Any]) -> None:
    """
    Print the response in a nicely formatted way
    
    Args:
        response: The response from the assistant
    """
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

def interactive_mode(assistant: EnhancedProductFeatureAssistant) -> None:
    """
    Run the assistant in interactive mode
    
    Args:
        assistant: The initialized assistant
    """
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
            response = process_query(assistant, query)
            print_response(response)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Fuzzy RAG Product Feature Assistant")
    
    parser.add_argument("--query", type=str, help="Process a single query")
    parser.add_argument("--dict-path", default="dictionaries", help="Path to dictionary files")
    parser.add_argument("--use-sample", action="store_true", help="Use sample dictionary data")
    parser.add_argument("--llm-model", default="llama3", help="LLM model to use")
    parser.add_argument("--llm-provider", default="mock", help="LLM provider (ollama, openai, anthropic, or mock)")
    
    args = parser.parse_args()
    
    # Create dictionaries directory if it doesn't exist
    os.makedirs(args.dict_path, exist_ok=True)
    
    # Set up the assistant
    assistant = setup_assistant(
        dictionaries_path=args.dict_path,
        use_sample_data=args.use_sample,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider
    )
    
    # Process a single query or run in interactive mode
    if args.query:
        response = process_query(assistant, args.query)
        print_response(response)
    else:
        interactive_mode(assistant)

if __name__ == "__main__":
    main()