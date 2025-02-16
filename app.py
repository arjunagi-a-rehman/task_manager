import streamlit as st
import boto3
import json
import uuid
from datetime import datetime
import pytz
from botocore.exceptions import ClientError
import os

def load_credentials():
    """Load AWS credentials and secret key from environment variables"""
    return {
        'aws_access_key': os.getenv('ACCESS_KEY_ID'),
        'aws_secret_key': os.getenv('SECRET_ACCESS_KEY'),
        'agent_id': os.getenv('BEDROCK_AGENT_ID'),
        'agent_alias': os.getenv('BEDROCK_AGENT_ALIAS'),
        'secret_key': os.getenv('APP_SECRET_KEY')  # Secret key for authentication
    }

def initialize_bedrock_client(credentials):
    """Initialize Bedrock client with credentials"""
    try:
        session = boto3.Session(
            aws_access_key_id=credentials['aws_access_key'],
            aws_secret_access_key=credentials['aws_secret_key'],
            region_name='us-east-1'
        )
        return session.client('bedrock-agent-runtime')
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return None

def get_bedrock_response(prompt):
    """Get response from Bedrock agent"""
    try:
        response = bedrock_agent.invoke_agent(
            agentId=credentials['agent_id'],
            agentAliasId=credentials['agent_alias'],
            sessionId=session_id,
            inputText=prompt
        )
        
        if 'completion' not in response:
            return "No completion in response"
            
        full_response = ""
        for event in response['completion']:
            try:
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

# Load credentials
credentials = load_credentials()
if not credentials:
    st.error("Failed to load credentials. Please check your environment variables.")
    st.stop()

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Authentication prompt
if not st.session_state.authenticated:
    st.title("Authentication Required 🔒")
    user_secret = st.text_input("Enter Secret Key:", type="password")
    if st.button("Submit"):
        if user_secret == credentials['secret_key']:
            st.session_state.authenticated = True
            st.success("Authentication successful! You may proceed.")
            st.rerun()  # Refresh the app to show chat UI
        else:
            st.error("Incorrect secret key. Try again.")
    st.stop()

# Initialize Bedrock Agent Runtime client
bedrock_agent = initialize_bedrock_client(credentials)
if not bedrock_agent:
    st.error("Failed to initialize Bedrock client")
    st.stop()

session_id = str(uuid.uuid4())  # Create a unique session ID

# Send current date and time to the agent (only once per session)
if 'datetime_sent' not in st.session_state:
    current_dt = datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    # Send a system update to the agent with the current date and time
    _ = get_bedrock_response(f"System update: The current date and time is {current_dt}.")
    st.session_state.datetime_sent = True

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Sidebar: Configuration & Info
# -------------------------
st.sidebar.header("Configuration")
st.sidebar.text(f"Session ID: {session_id}")

st.sidebar.markdown("---")
st.sidebar.markdown("## About This Chatbot")
st.sidebar.markdown("""
**Adaptive Task Manager**

I am an Adaptive Task Manager that helps you organize, track, and plan your work using existing API capabilities. I maintain full context of your ongoing tasks, including priorities, dependencies, and progress. When needed, I intelligently interpret your queries and may delegate external questions to a Web Search Agent.

**Rail Guard:**  
If you ask a question that appears off-topic (for example, about celebrities or general non-work matters), I will ask a clarifying question such as:  
_"I'm here to help manage your tasks and projects. Could you please explain how that question relates to your work or tasks?"_
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## How to Use")
st.sidebar.markdown("""
1. **Authentication:**  
   - Enter the secret key provided (from your environment variables) to access the chatbot.
   
2. **Task Management:**  
   - Type your task-related queries or commands (e.g., "Create a new task: Review project proposal", "Show my pending tasks", "Mark task #123 as complete").
   - I will help organize and manage your tasks based on your input.

3. **General Guidance:**  
   - If your query doesn’t seem related to work or task management, I may ask for clarification on how it pertains to your projects.
""")

# -------------------------
# Main UI: Chat Interface
# -------------------------
st.title("Task Management Chatbot")

# Display chat conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What task would you like to manage?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_bedrock_response(prompt)
            if response:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# -------------------------
# Sidebar: Debugging Information
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Debug Information")
if st.sidebar.button("Check AWS Connection"):
    try:
        test_response = bedrock_agent.invoke_agent(
            agentId=credentials['agent_id'],
            agentAliasId=credentials['agent_alias'],
            sessionId=session_id,
            inputText='test'
        )
        st.sidebar.success("AWS Connection Successful!")
    except Exception as e:
        st.sidebar.error(f"AWS Connection Error: {str(e)}")
