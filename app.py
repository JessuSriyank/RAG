import os
import time
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load environment variables
groq_api_key = "Your_api_key"

# Set up Streamlit
st.title("Simple RAG Application")

def hide_streamlit_style():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

hide_streamlit_style()

# Initialize the language model
print("Initializing ChatGroq model...")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
print("ChatGroq model initialized.")

# File upload for PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# Process PDF and create vector store
if uploaded_file:
    print("PDF file uploaded. Saving to disk...")
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    print("PDF saved as 'uploaded_document.pdf'.")

    # Initialize embeddings and document loader
    try:
        print("Initializing Hugging Face embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Hugging Face embeddings initialized.")
    except Exception as e:
        st.error(f"Failed to initialize embeddings: {e}")
        print(f"Error initializing embeddings: {e}")
        embeddings = None

    if embeddings:
        print("Loading and processing the PDF...")
        loader = PyPDFLoader("uploaded_document.pdf")
        docs = loader.load()
        print(f"Loaded {len(docs)} documents from PDF.")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        final_documents = text_splitter.split_documents(docs[:20])
        print(f"Split documents into {len(final_documents)} chunks.")

        # Create FAISS vector store
        try:
            print("Creating FAISS vector store...")
            vectors = FAISS.from_documents(final_documents, embeddings)
            print("FAISS vector store created successfully.")
        except Exception as e:
            vectors = None
            print(f"Error during FAISS vector store creation: {e}")
            st.error(f"Failed to embed documents: {e}")

        if vectors:
            try:
                print("Setting up retriever...")
                retriever = vectors.as_retriever()
                print("Retriever setup complete.")

                # Define retrieval prompt template
                print("Defining prompt template...")
                prompt_template = ChatPromptTemplate.from_template(
                    """
                    Answer the question based on the provided context only.
                    Please provide the most accurate response based on the question.
                    <context>
                    {context}
                    <context>
                    Question: {input}
                    """
                )
                print("Prompt template defined.")

                # Set up document chain and retrieval chain
                print("Setting up chains...")
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                print("Chains set up successfully.")

                # Get user query
                user_query = st.text_input("Enter your question here:")
                if user_query:
                    print(f"User query received: {user_query}")
                    start = time.process_time()
                    try:
                        response = retrieval_chain.invoke({"input": user_query})
                        end_time = time.process_time() - start
                        print(f"Response received in {end_time:.2f} seconds.")
                        st.write(f"Response time: {end_time:.2f} seconds")
                        st.subheader("AI Response:")
                        st.write(response['answer'])

                        # Display retrieved documents with similarity search
                        with st.expander("Document Similarity Search Results"):
                            for i, doc in enumerate(response["context"]):
                                print(f"Displaying result {i + 1}")
                                st.write(f"**Result {i + 1}:**")
                                st.write(doc.page_content)
                                st.write("--------------------------------")
                    except Exception as e:
                        print(f"Error during query processing: {e}")
                        st.error(f"Failed to process the query: {e}")
            except Exception as e:
                print(f"Error setting up retriever or chains: {e}")
                st.error(f"Could not complete the setup: {e}")
        else:
            print("Skipping retriever and chain setup due to FAISS failure.")
else:
    st.warning("Please upload a PDF file to start.")
    print("No PDF uploaded. Waiting for user input.")
