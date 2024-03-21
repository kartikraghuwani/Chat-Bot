from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from docx import Document
from langchain_core.messages import AIMessage, HumanMessage


# For PDF Files
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# For Docx files
def get_word_text(docx_file):
    document = Document(docx_file)
    text = "\n".join([paragraph.text for paragraph in document.paragraphs])
    return text

# For Txt files
def read_text_file(txt_file):
    text = txt_file.getvalue().decode('utf-8')
    return text

#combining the text
def combine_text(text_list):
    return "\n".join(text_list)

# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
  if st.session_state.conversation is not None:
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

def main():
    load_dotenv()
    st.set_page_config(
        page_title="File Chatbot",
        page_icon=":books:",
        layout="wide"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]   

    st.header("Chat with your multiple files:")

    # Initialize variables to hold uploaded files
    other_files = []

    with st.sidebar:
        
        st.subheader("Your documents")
        files = st.file_uploader(
            "Upload your files here and click on 'Process'", accept_multiple_files=True)
        
        for file in files:
            if file.name.lower().endswith('.csv'):
                csv_file = file  # Store the CSV file

            else:
                other_files.append(file)  # Store other file types

        # Initialize empty lists for each file type
        pdf_texts = []
        word_texts = []
        txt_texts = []

        if st.button("Process"):
            with st.spinner("Processing"):
                for file in other_files:
                    if file.name.lower().endswith('.pdf'):
                        pdf_texts.append(get_pdf_text(file))
                    elif file.name.lower().endswith('.docx'):
                        word_texts.append(get_word_text(file))
                    elif file.name.lower().endswith('.txt'):
                        txt_texts.append(read_text_file(file))

                # Combine text from different file types
                combined_text = combine_text(pdf_texts + word_texts + txt_texts)

                # Split the combined text into chunks
                text_chunks = get_text_chunks(combined_text)

                # Create vector store and conversation chain if non-CSV documents are uploaded
                if len(other_files) > 0:
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                else:
                    vectorstore = None  # No need for vectorstore with CSV file

    user_query = st.chat_input("Type your message here...")

    # Handle user input for text-based files
    if user_query:
        handle_user_input(user_query)

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

if __name__ == '__main__':
    main()