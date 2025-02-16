import streamlit as st
import boto3
import json
import uuid
from datetime import datetime
import pytz
from botocore.exceptions import ClientError
import os

def load_credentials():
    """Load AWS credentials from environment variables"""
    return {
        'aws_access_key': os.getenv('ACCESS_KEY_ID'),
        'aws_secret_key': os.getenv('SECRET_ACCESS_KEY'),
        'agent_id': os.getenv('BEDROCK_AGENT_ID'),
        'agent_alias': os.getenv('BEDROCK_AGENT_ALIAS')
    }

def initialize_bedrock_client(credentials):
    """Initialize Bedrock client with credentials"""
    try:
        # Create a session with the credentials
        session = boto3.Session(
            aws_access_key_id=credentials['aws_access_key'],
            aws_secret_access_key=credentials['aws_secret_key'],
            region_name='us-east-1'
        )
        
        # Create the bedrock-agent-runtime client
        return session.client('bedrock-agent-runtime')
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return None

# Load credentials
credentials = load_credentials()
if not credentials:
    st.error("Failed to load credentials. Please check your credentials.yml file.")
    st.stop()

# Initialize Bedrock Agent Runtime client
bedrock_agent = initialize_bedrock_client(credentials)
if not bedrock_agent:
    st.error("Failed to initialize Bedrock client")
    st.stop()

session_id = str(uuid.uuid4())  # Create a unique session ID

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_bedrock_response(prompt):
    """
    Get response from Bedrock agent
    """
    try:
        response = bedrock_agent.invoke_agent(
            agentId=credentials['agent_id'],
            agentAliasId=credentials['agent_alias'],
            sessionId=session_id,
            inputText=prompt
        )
        
        # Debug information
        st.sidebar.markdown("### Debug Info")
        st.sidebar.text("Response keys: " + str(response.keys()))
        
        if 'completion' not in response:
            return "No completion in response"
            
        # Handle the streaming response
        full_response = ""
        
        for event in response['completion']:
            try:
                # Extract bytes from the nested structure
                if isinstance(event, dict) and 'chunk' in event:
                    chunk = event['chunk']
                    if isinstance(chunk, dict) and 'bytes' in chunk:
                        bytes_data = chunk['bytes']
                        if isinstance(bytes_data, bytes):
                            text = bytes_data.decode('utf-8')
                            full_response += text
                    
            except Exception as e:
                st.error(f"Error processing chunk: {str(e)}")
                continue
                
        return full_response.strip()
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        st.error(f"AWS Error: {error_code} - {error_message}")
        return None
    except Exception as e:
        st.error(f"Error communicating with Bedrock: {str(e)}")
        return None

# Streamlit UI
st.title("Task Management Chatbot")

# Display configuration from YAML
st.sidebar.header("Configuration")
st.sidebar.text(f"Agent ID: {credentials['agent_id']}")
st.sidebar.text(f"Agent Alias: {credentials['agent_alias']}")
st.sidebar.text(f"Session ID: {session_id}")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What task would you like to manage?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_bedrock_response(prompt)
            if response:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Add helpful information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to use:
1. Ensure credentials.yml is in the same directory
2. Type your task-related question or command
3. The chatbot will help you manage your tasks

### Example commands:
- "Create a new task: Review project proposal"
- "Show my pending tasks"
- "Mark task #123 as complete"
- "What's my next deadline?"
- "help me with following tasks"
- "How to finish following task"
""")

# Add debugging information
st.sidebar.markdown("---")
st.sidebar.markdown("### Debug Information")
if st.sidebar.button("Check AWS Connection"):
    try:
        # Test the connection
        test_response = bedrock_agent.invoke_agent(
            agentId=credentials['agent_id'],
            agentAliasId=credentials['agent_alias'],
            sessionId=session_id,
            inputText='test'
        )
        st.sidebar.success("AWS Connection Successful!")
    except Exception as e:
        st.sidebar.error(f"AWS Connection Error: {str(e)}")