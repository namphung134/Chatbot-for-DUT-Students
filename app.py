import streamlit as st
from path import *
from pathlib import Path


def main():
    st.set_page_config(
        page_title="DUT University Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        # pdf_docs = st.file_uploader(
        #     "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        # if st.button("Submit & Process"):
        #     with st.spinner("Processing..."):
        #         raw_text = get_pdf_text(pdf_docs)
        #         text_chunks = get_text_chunks(raw_text)
        #         get_vector_store(text_chunks)
        #         st.success("Done")
    
    pdf_docs = Path("data").glob("*.pdf")
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
                
    # Main content area for displaying chat messages
    st.title("Chatbot for DUT students about DUT university regulations🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
