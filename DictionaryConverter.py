#!/usr/bin/env python3
"""
Utility module to convert from the existing dictionary format to the new format
This helps integrate with your existing FuzzyProductSearchParser implementation
"""

import os
import json
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DictionaryConverter:
    """
    Utility class to convert dictionaries between different formats
    """
    
    @staticmethod
    def convert_features_dictionary(input_path: str, output_path: str) -> bool:
        """
        Convert a features dictionary from the existing format to the new format
        
        Args:
            input_path: Path to the input dictionary file
            output_path: Path to save the converted dictionary
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Converting features dictionary from {input_path} to {output_path}")
        
        try:
            # Read the input dictionary
            with open(input_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
                
            # Convert to the new format
            output_data = []
            
            for item in input_data:
                # Check if this is in the new format already (has "feature" key)
                if "feature" in item:
                    output_data.append(item)
                    continue
                    
                # Convert from old format
                feature_name = item.get("name", "")
                if not feature_name:
                    continue
                    
                new_feature = {
                    "feature": feature_name,
                    "description": item.get("description", ""),
                    "spelling_variations": item.get("spelling_variations", []),
                    "synonyms": item.get("synonyms", [])
                }
                
                # Add optional fields if present
                if "benefits" in item:
                    new_feature["benefits"] = item["benefits"]
                    
                if "technical_specs" in item:
                    new_feature["technical_specs"] = item["technical_specs"]
                    
                output_data.append(new_feature)
                
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
            # Write the output dictionary
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
                
            logger.info(f"Successfully converted {len(output_data)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error converting features dictionary: {e}")
            return False
    
    @staticmethod
    def convert_from_existing_structure(input_dict: Dict[str, Any], feature_key: str = "feature") -> List[Dict[str, Any]]:
        """
        Convert from your existing dictionary structure to the new list-based structure
        
        Args:
            input_dict: The input dictionary in your existing format
            feature_key: The key to use for the feature name
            
        Returns:
            List of feature dictionaries in the new format
        """
        output_list = []
        
        # Process the "fullCategoryData" structure from your existing code
        if "fullCategoryData" in input_dict:
            categories = input_dict["fullCategoryData"]
            
            # Extract features data
            if "features" in categories:
                for feature in categories["features"]:
                    output_list.append(feature)
        
        return output_list
    
    @staticmethod
    def generate_sample_dictionaries(output_dir: str = "dictionaries") -> bool:
        """
        Generate sample dictionary files for testing
        
        Args:
            output_dir: Directory to save the sample dictionaries
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Generating sample dictionaries in {output_dir}")
        
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
                
            # Sample product types
            product_types = [
                {
                    "type": "Chair",
                    "description": "A seat with a back and often with arms",
                    "spelling_variations": ["chairs", "seat"],
                    "synonyms": ["seating", "seat"]
                },
                {
                    "type": "Desk",
                    "description": "A piece of furniture with a flat or sloped surface",
                    "spelling_variations": ["desks", "table"],
                    "synonyms": ["workstation", "table"]
                },
                {
                    "type": "Bookshelf",
                    "description": "A piece of furniture with shelves for storing books",
                    "spelling_variations": ["bookshelves", "book shelf"],
                    "synonyms": ["shelving", "shelf", "bookcase"]
                }
            ]
            
            # Write the product types dictionary
            with open(os.path.join(output_dir, "productTypeDictionary.json"), 'w', encoding='utf-8') as f:
                json.dump(product_types, f, indent=2)
                
            # Sample styles
            styles = [
                {
                    "name": "Modern",
                    "description": "Clean lines and minimalist aesthetic",
                    "spelling_variations": ["modernist", "contemporary"],
                    "synonyms": ["contemporary", "current", "present-day"]
                },
                {
                    "name": "Industrial",
                    "description": "Raw materials and utilitarian design",
                    "spelling_variations": ["industrial style", "factory style"],
                    "synonyms": ["factory", "warehouse", "utility"]
                },
                {
                    "name": "Scandinavian",
                    "description": "Simple, clean design with natural materials",
                    "spelling_variations": ["nordic", "scandi"],
                    "synonyms": ["Nordic", "minimalist", "natural"]
                }
            ]
            
            # Write the styles dictionary
            with open(os.path.join(output_dir, "stylesDictionary.json"), 'w', encoding='utf-8') as f:
                json.dump(styles, f, indent=2)
                
            logger.info("Successfully generated sample dictionaries")
            return True
            
        except Exception as e:
            logger.error(f"Error generating sample dictionaries: {e}")
            return False


if __name__ == "__main__":
    # Test the converter
    converter = DictionaryConverter()
    
    # Generate sample dictionaries
    success = converter.generate_sample_dictionaries("test_dictionaries")
    print(f"Generated sample dictionaries: {success}")
    
    # Convert an old format dictionary
    old_format = [
        {
            "name": "Comfortable Padding",
            "description": "Extra padding for comfort",
            "spelling_variations": ["comfort padding", "extra padding"],
            "synonyms": ["cushioning", "soft padding"]
        },
        {
            "name": "Breathable Fabric",
            "description": "Fabric that allows air circulation",
            "spelling_variations": ["breathing fabric", "breatheable"],
            "synonyms": ["ventilated fabric", "air-flow"]
        }
    ]
    
    # Save old format
    os.makedirs("test_conversion", exist_ok=True)
    with open("test_conversion/old_format.json", 'w', encoding='utf-8') as f:
        json.dump(old_format, f, indent=2)
        
    # Convert to new format
    success = converter.convert_features_dictionary(
        "test_conversion/old_format.json",
        "test_conversion/new_format.json"
    )
    print(f"Converted old format to new format: {success}")
    
    # Read the converted file
    with open("test_conversion/new_format.json", 'r', encoding='utf-8') as f:
        new_format = json.load(f)
        
    print("New format sample:")
    print(json.dumps(new_format[0], indent=2))