import openai
import streamlit as st


st.title = "CLONE"
openai.api_key = st.secrets["OPEN_API_KEY"]
print(st.secrets["OPEN_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

#Initilaize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What's up?")
if prompt:
    with st.chat_message("user"):
         st.markdown(prompt)
    st.session_state.messages.append({"role" : "user", "content": prompt})
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.chat.completions.create(
            model = st.session_state["openai_model"],
            messages = [
            { "role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            ],
            stream = True
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "🙈")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    
        

