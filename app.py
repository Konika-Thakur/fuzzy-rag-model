#!/usr/bin/env python3
"""
Streamlit web interface for the Enhanced Fuzzy RAG Product Feature Assistant
"""

import os
import json
import logging
import streamlit as st
from typing import Dict, List, Any

from config import Config
from main_assistant import EnhancedProductFeatureAssistant

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

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'history' not in st.session_state:
    st.session_state.history = []

def initialize_system(
    dictionaries_path: str,
    llm_provider: str,
    llm_model: str,
    api_key: str = None
) -> EnhancedProductFeatureAssistant:
    """Initialize the product feature assistant"""
    try:
        # Create the assistant
        assistant = EnhancedProductFeatureAssistant(
            dictionaries_path=dictionaries_path,
            llm_provider=llm_provider,
            llm_model=llm_model,
            api_key=api_key
        )
        
        # Initialize system
        success = assistant.initialize_from_dictionaries()
        
        if not success:
            st.error("Failed to initialize the system")
            return None
            
        return assistant
        
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        logger.error(f"Error initializing system: {e}")
        return None

def display_response(response: Dict[str, Any]) -> None:
    """Display the response in a nicely formatted way for multi-dictionary results"""
    # Display corrected query
    if response['corrected_query'] != response['original_query']:
        st.info(f"**Interpreted as:** {response['corrected_query']}")
    
    # Display identified entities by type
    st.subheader("🎯 Identified Entities")
    
    entity_type_icons = {
        'features': '⚙️',
        'styles': '🎨', 
        'products': '🪑',
        'places': '🏠'
    }
    
    entity_type_names = {
        'features': 'Features',
        'styles': 'Styles',
        'products': 'Products', 
        'places': 'Places'
    }
    
    # Create columns for different entity types
    cols = st.columns(len([entities for entities in response['identified_entities'].values() if entities]))
    col_idx = 0
    
    for entity_type, entities in response['identified_entities'].items():
        if not entities:
            continue
            
        with cols[col_idx]:
            icon = entity_type_icons.get(entity_type, '📋')
            type_name = entity_type_names.get(entity_type, entity_type.title())
            st.markdown(f"**{icon} {type_name}**")
            
            for entity in entities:
                confidence_color = "🟢" if entity['confidence'] > 0.8 else "🟡" if entity['confidence'] > 0.6 else "🔴"
                st.write(f"{confidence_color} {entity['name']} ({entity['confidence']:.2f})")
        
        col_idx += 1
    
    # Display explanation
    st.subheader("📝 Explanation")
    st.markdown(response['explanation'])
    
    # Display similar entities
    if response['similar_entities']:
        st.subheader("💡 Related Items You Might Like")
        
        # Flatten similar entities for display
        all_similar = []
        for entity_type, entity_similarities in response['similar_entities'].items():
            for entity_name, similar_list in entity_similarities.items():
                for similar in similar_list:
                    all_similar.append({
                        'original': entity_name,
                        'original_type': entity_type,
                        **similar
                    })
        
        # Group by type for organized display
        similar_by_type = {}
        for similar in all_similar:
            similar_type = similar.get('type', 'unknown')
            if similar_type not in similar_by_type:
                similar_by_type[similar_type] = []
            similar_by_type[similar_type].append(similar)
        
        # Display in columns by type
        if similar_by_type:
            similar_cols = st.columns(len(similar_by_type))
            for idx, (similar_type, similar_items) in enumerate(similar_by_type.items()):
                with similar_cols[idx]:
                    icon = entity_type_icons.get(similar_type, '📋')
                    type_name = entity_type_names.get(similar_type, similar_type.title())
                    st.markdown(f"**{icon} {type_name}**")
                    
                    for similar in similar_items[:3]:  # Limit to top 3 per type
                        similarity_score = similar.get('similarity', 0)
                        similarity_bar = "█" * max(1, int(similarity_score * 5)) if similarity_score > 0 else ""
                        st.write(f"• **{similar['name']}** {similarity_bar}")
                        if similar.get('description'):
                            st.caption(similar['description'][:80] + "..." if len(similar['description']) > 80 else similar['description'])

def display_entity_library() -> None:
    """Display the available entities library for all dictionary types"""
    if st.session_state.assistant and st.session_state.initialized:
        all_entities = st.session_state.assistant.get_available_entities()
        
        if all_entities:
            with st.expander("📚 Available Entities Library", expanded=False):
                # Create tabs for different entity types
                entity_type_icons = {
                    'features': '⚙️ Features',
                    'styles': '🎨 Styles', 
                    'products': '🪑 Products',
                    'places': '🏠 Places'
                }
                
                tabs = st.tabs([entity_type_icons.get(etype, etype.title()) for etype in all_entities.keys() if all_entities[etype]])
                
                tab_idx = 0
                for entity_type, entities in all_entities.items():
                    if not entities:
                        continue
                        
                    with tabs[tab_idx]:
                        for entity in entities:
                            # Get entity name based on type
                            if entity_type == 'features':
                                entity_name = entity.get('feature', '')
                            elif entity_type in ['styles', 'places']:
                                entity_name = entity.get('name', '')
                            elif entity_type == 'products':
                                entity_name = entity.get('type', '')
                            else:
                                entity_name = entity.get('name', '')
                            
                            description = entity.get('description', '')
                            
                            if entity_name:
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.markdown(f"**{entity_name}**")
                                
                                with col2:
                                    st.write(description)
                                    
                                    # Display variations and synonyms
                                    variations = entity.get('spelling_variations', [])
                                    synonyms = entity.get('synonyms', [])
                                    
                                    if variations or synonyms:
                                        all_variations = variations + synonyms
                                        st.caption(f"Also matches: {', '.join(all_variations[:5])}")
                                
                                st.divider()
                    
                    tab_idx += 1

def display_system_status() -> None:
    """Display system status information for multi-dictionary system"""
    if st.session_state.assistant:
        status = st.session_state.assistant.get_system_status()
        
        st.subheader("🔧 System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("System Status", "✅ Ready" if status['initialized'] else "❌ Not Ready")
            if 'total_entities' in status:
                st.metric("Total Entities", status['total_entities'])
        
        with col2:
            st.metric("LLM Provider", status.get('llm_provider', 'Unknown'))
            st.metric("Vector Storage", status.get('vector_storage_type', 'Unknown'))
        
        with col3:
            st.metric("Embeddings", status.get('embeddings_count', 0))
            st.metric("Qdrant Connected", "✅" if status.get('qdrant_connected', False) else "❌")
        
        # Display entity counts by type
        if 'entity_counts' in status:
            st.subheader("📊 Entity Counts by Type")
            entity_cols = st.columns(len(status['entity_counts']))
            
            entity_type_icons = {
                'features': '⚙️',
                'styles': '🎨', 
                'products': '🪑',
                'places': '🏠'
            }
            
            for idx, (entity_type, count) in enumerate(status['entity_counts'].items()):
                with entity_cols[idx]:
                    icon = entity_type_icons.get(entity_type, '📋')
                    st.metric(f"{icon} {entity_type.title()}", count)

def sidebar_configuration():
    """Handle sidebar configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dictionary settings
        st.subheader("📁 Dictionary Settings")
        dictionaries_path = st.text_input("Dictionaries Path", value=Config.DICTIONARIES_PATH)
        
        # LLM settings
        st.subheader("🤖 LLM Settings")
        llm_provider = st.selectbox(
            "LLM Provider",
            ["mock", "ollama", "openai", "anthropic"],
            index=0
        )
        
        llm_model = st.text_input("LLM Model", value=Config.LLM_MODEL)
        
        api_key = None
        if llm_provider in ["openai", "anthropic"]:
            api_key = st.text_input("API Key", type="password")
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing system..."):
                st.session_state.assistant = initialize_system(
                    dictionaries_path=dictionaries_path,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    api_key=api_key
                )
                
                if st.session_state.assistant:
                    st.session_state.initialized = True
                    st.success("✅ System initialized successfully!")
                    st.rerun()
        
        # Display system status
        display_system_status()
        
        # Example queries
        st.subheader("💬 Example Queries")
        example_queries = [
            "modern chair with lumbar support for my office",
            "scandinavian desk with adjustable height", 
            "industrial style sofa for living room",
            "ergonomic chair with metal legs",
            "contemporary bedroom furniture with wooden sleek arms",
            "office chair with swivel base and high back"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{hash(query)}"):
                if st.session_state.initialized:
                    st.session_state.current_query = query
                    st.rerun()
                else:
                    st.warning("⚠️ Please initialize the system first")

def main():
    """Main function for the Streamlit app"""
    st.title("🔍 Enhanced Fuzzy RAG Product Feature Assistant")
    st.markdown(
        """
        This system processes natural language queries that may contain **typos** or **non-standard terminology**, 
        identifies product features, and provides comprehensive explanations with similar feature suggestions.
        """
    )
    
    # Sidebar configuration
    sidebar_configuration()
    
    # Main content area
    if st.session_state.initialized and st.session_state.assistant:
        # Display feature library
        display_entity_library()
        
        # Query input section
        st.subheader("💭 Enter Your Query")
        
        # Handle example query from sidebar
        default_query = st.session_state.get("current_query", "")
        if default_query:
            st.session_state.current_query = ""  # Clear it after using
        
        query = st.text_input(
            "What are you looking for?",
            value=default_query,
            placeholder="e.g., modern chair with lumbar support for my office",
            help="Try including styles, products, features, and places - the system understands them all!"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            process_button = st.button("🔍 Process Query", type="primary")
        
        with col2:
            clear_history = st.button("🗑️ Clear History")
            if clear_history:
                st.session_state.history = []
                st.rerun()
        
        # Process query
        if query and process_button:
            with st.spinner("🔄 Processing your query..."):
                try:
                    response = st.session_state.assistant.process_query(query)
                    
                    # Add to history
                    st.session_state.history.append({
                        "query": query,
                        "response": response
                    })
                    
                    # Display response
                    st.subheader("📋 Response")
                    display_response(response)
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
                    logger.error(f"Error processing query: {e}")
        
        # Display history
        if st.session_state.history:
            st.subheader("📚 Query History")
            
            # Show only last 3 items to avoid clutter
            for i, item in enumerate(reversed(st.session_state.history[-3:])):
                with st.expander(f"Query {len(st.session_state.history) - i}: {item['query'][:50]}..."):
                    display_response(item['response'])
    
    else:
        # System not initialized
        st.info("👈 Please initialize the system using the sidebar controls to get started.")
        
        # Display information about the system
        st.subheader("🚀 Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Key Features:**
            - 🔤 **Multi-Dictionary Support**: Handles features, styles, products, and places
            - 🔄 **Fuzzy Matching**: Corrects typos and understands variations
            - 🧠 **AI-Powered**: Uses LLM for comprehensive explanations
            - 🔍 **Vector Search**: Finds similar items using embeddings
            - 📊 **Confidence Scoring**: Shows certainty levels
            """)
        
        with col2:
            st.markdown("""
            **Entity Types Available:**
            - ⚙️ **Features**: Lumbar support, adjustable height, metal legs
            - 🎨 **Styles**: Modern, industrial, scandinavian
            - 🪑 **Products**: Chair, desk, sofa, table
            - 🏠 **Places**: Office, living room, bedroom
            """)
        
        st.markdown("""
        **How to Use:**
        1. Configure Ollama with llama3.2:3b in the sidebar
        2. Click "Initialize System" to load all dictionaries
        3. Enter queries combining multiple entity types
        4. Get detailed explanations showing how everything works together
        
        **Example Query:** "modern chair with lumbar support for my office"
        - 🎨 Style: Modern
        - 🪑 Product: Chair  
        - ⚙️ Feature: Lumbar Support
        - 🏠 Place: Office
        """)

if __name__ == "__main__":
    main()