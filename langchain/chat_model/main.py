from langchain_groq import ChatGroq
from dotenv import load_dotenv 
import streamlit as st
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

st.sidebar.header("Chat with llama-3.1-70b")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if not message['role']=='system':
        with st.chat_message(message['role']):
            st.write(message['content'])
    else:
        st.header(message['content'])

message = st.chat_input("Chat with it")
context = st.sidebar.chat_input("Context")

if context:
    st.session_state.messages.clear()
    st.session_state.messages.insert(0,{"role": "system","content" : context})
    st.subheader(context)

if message :
    with st.chat_message("User"):
        st.write(message)
    st.session_state.messages.append({"role": "user","content" : message})
    ai_msg = llm.invoke(st.session_state.messages)
    with st.chat_message("AI"):
        st.write(ai_msg.content)
    st.session_state.messages.append({"role": "ai","content" : ai_msg.content})