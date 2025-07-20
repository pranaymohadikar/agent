# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Configuration
FASTAPI_URL = "http://localhost:8000"  # Update if your API is hosted elsewhere
TITLE = "📊 Natural Language to SQL Query Generator"

# Set page config
st.set_page_config(
    page_title=TITLE,
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stTextArea textarea {
        min-height: 150px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stMarkdown {
        padding: 10px;
        border-radius: 5px;
    }
    pre {
        white-space: pre-wrap;
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def call_fastapi_endpoint(question: str):
    """Call the FastAPI /query endpoint"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/query",
            json={"question": question},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def display_results(data):
    """Display the results in a user-friendly format"""
    if not data:
        return
    
    st.subheader("Generated SQL Query")
    st.code(data["sql_query"], language="sql")
    
    st.subheader("Query Results")
    if isinstance(data["result"], list) and data["result"]:
        st.dataframe(pd.DataFrame(data["result"]))
        st.dataframe(data['result'])
    else:
        st.info("No results returned from the query")

def show_api_info():
    """Display API information in the sidebar"""
    with st.sidebar:
        st.header("API Information")
        
        # API Status Check
        if st.button("Check API Status"):
            try:
                response = requests.get(f"{FASTAPI_URL}/")
                if response.status_code == 200:
                    st.success("API is running and reachable")
                else:
                    st.error(f"API returned status code: {response.status_code}")
            except:
                st.error("API is not reachable")
        
        # Schema Viewer
        if st.button("View Database Schema"):
            try:
                response = requests.get(f"{FASTAPI_URL}/schema")
                st.text_area("Database Schema", 
                           response.json()["schema"], 
                           height=300)
            except Exception as e:
                st.error(f"Couldn't fetch schema: {str(e)}")

def main():
    st.title(TITLE)
    st.markdown("Ask questions in natural language and get SQL query results")
    
    # Show API info in sidebar
    show_api_info()
    
    # Main input area
    question = st.text_area(
        "Enter your question about the data:",
        placeholder="e.g., 'Show all messages from Mumbai with positive sentiment' or 'Count messages from May 2024'",
        height=150
    )
    
    # Process question when button is clicked
    if st.button("Generate SQL and Execute"):
        if not question.strip():
            st.warning("Please enter a question")
            return
        
        with st.spinner("Generating SQL and executing query..."):
            start_time = datetime.now()
            result = call_fastapi_endpoint(question)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if result:
                st.success(f"Query processed in {processing_time:.2f} seconds")
                display_results(result)
                
                # Debug view (can be toggled)
                with st.expander("Raw API Response"):
                    st.json(result)

if __name__ == "__main__":
    main()