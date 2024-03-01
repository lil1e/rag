
import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from PyPDF2 import PdfReader
from langchain.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import tongyi
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dashscope
import os
import time
import jieba
os.environ["DASHSCOPE_API_KEY"]="sk-d139c49f73de48d5a9c7c86a6fd2db23"
dashscope.api_key=os.environ["DASHSCOPE_API_KEY"]
#langchian == 0.0.354

#----------------------------------------------------------------------------

# Extracts and concatenates text from a list of PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
 
# Splits a given text into smaller chunks based on specified conditions
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
# Generates embeddings for given text chunks and creates a vector store using FAISS
def get_vectorstore(text_chunks):
    embeddings = DashScopeEmbeddings(
    model="text-embedding-v1"
)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Initializes a conversation chain with a given vector store
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=tongyi.ChatTongyi(model_name='qwen-7b-chat'),
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h: h,
        memory=memory
    )
    return conversation_chain

def stream_data(content):
    seg_list = jieba.cut(content, cut_all=False)
    for word in seg_list:
        yield word
        time.sleep(0.02)

#-----------------------------------------------------------------------


def run():
  st.set_page_config(page_title="Chat with your assistant", layout="wide")

  st.title("🤠 Chat with your assistant")
  st.header("Please upload relevant information first") 

  user_uploads = st.file_uploader("Upload your files", accept_multiple_files=True)
  if user_uploads is not None:
      if st.button("Upload"):
          with st.spinner("Processing"):
              # Get PDF Text
              raw_text = get_pdf_text(user_uploads)
              # Retrieve chunks from text
              text_chunks = get_text_chunks(raw_text)
              # Create FAISS Vector Store of PDF Docs
              vectorstore = get_vectorstore(text_chunks)
              # Create conversation chain
              st.session_state.conversation = get_conversation_chain(vectorstore)

  

  #调试信息
  #st.write(st.session_state)  
  if st.session_state.get('conversation') is not None:
        # st.session_state["messages"] = st.session_state.conversation.memory.chat_memory.messages
        for message in st.session_state.conversation.memory.chat_memory.messages:
                    if isinstance(message, HumanMessage):
                        with st.chat_message("user"):
                            st.markdown(message.content)
                    elif isinstance(message, AIMessage):
                        with st.chat_message("assistant"):
                            st.markdown(message.content)


  if user_query := st.chat_input("Enter your query here"):
      stream = ""
      with st.chat_message("user"):
          st.write(user_query)
      # Process the user's message using the conversation chain
      if 'conversation' in st.session_state:
          result = st.session_state.conversation({
              "question": user_query, 
              "chat_history": st.session_state.get('chat_history', [])
          })
          response = result["answer"]
          stream=stream_data(response)
      else:
          response = "Please upload a document first to initialize the conversation chain."
       
      with st.chat_message("assistant"):
          st.write_stream(stream)


if __name__ == "__main__":
    run()
