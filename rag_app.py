"""
RAG (Retrieval-Augmented Generation) Application using LangChain

This application demonstrates how to build a question-answering system
that can answer questions about the contents of a text file using
LangChain's RAG capabilities.
"""

import os
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class RAGPipeline:
    """
    A Retrieval-Augmented Generation pipeline for question answering.
    
    This class handles document loading, text splitting, embedding creation,
    vector store management, and question answering using LangChain.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            openai_api_key: OpenAI API key. If not provided, will look for
                          OPENAI_API_KEY environment variable.
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either as parameter or "
                "through OPENAI_API_KEY environment variable"
            )
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=self.api_key
        )
        self.vectorstore = None
        self.qa_chain = None
    
    def load_document(self, file_path: str) -> list:
        """
        Load a text document from file.
        
        Args:
            file_path: Path to the text file to load.
            
        Returns:
            List of document objects.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s) from {file_path}")
        return documents
    
    def split_documents(self, documents: list, chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> list:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of documents to split.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Number of characters to overlap between chunks.
            
        Returns:
            List of document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunk(s)")
        return chunks
    
    def create_vectorstore(self, documents: list) -> None:
        """
        Create a vector store from documents using embeddings.
        
        Args:
            documents: List of document chunks to embed and store.
        """
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        print("Vector store created successfully")
    
    def setup_qa_chain(self) -> None:
        """
        Set up the question-answering chain with retrieval.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        # Create a custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("QA chain set up successfully")
    
    def ask_question(self, question: str) -> dict:
        """
        Ask a question about the loaded documents.
        
        Args:
            question: The question to ask.
            
        Returns:
            Dictionary containing the answer and source documents.
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_qa_chain first.")
        
        result = self.qa_chain.invoke({"query": question})
        return {
            "question": question,
            "answer": result["result"],
            "source_documents": result.get("source_documents", [])
        }
    
    def initialize_from_file(self, file_path: str, chunk_size: int = 1000,
                            chunk_overlap: int = 200) -> None:
        """
        Convenience method to initialize the entire pipeline from a file.
        
        Args:
            file_path: Path to the text file to load.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        documents = self.load_document(file_path)
        chunks = self.split_documents(documents, chunk_size, chunk_overlap)
        self.create_vectorstore(chunks)
        self.setup_qa_chain()
        print(f"\nRAG pipeline initialized successfully!")
        print(f"Ready to answer questions about: {file_path}\n")


def main():
    """
    Main function demonstrating the RAG pipeline usage.
    """
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        print("\nFor demonstration purposes, showing how to use the pipeline:")
        print("\nExample usage:")
        print("=" * 60)
        print("""
from rag_app import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline(openai_api_key="your-api-key-here")

# Load and process a document
rag.initialize_from_file("sample_data.txt")

# Ask questions
result = rag.ask_question("What is LangChain?")
print(f"Q: {result['question']}")
print(f"A: {result['answer']}")

# Ask another question
result = rag.ask_question("What are the key features of LangChain?")
print(f"Q: {result['question']}")
print(f"A: {result['answer']}")
""")
        return
    
    # Initialize the RAG pipeline
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline()
    
    # Use the sample data file
    file_path = "sample_data.txt"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        print("Please make sure sample_data.txt exists in the current directory.")
        return
    
    # Initialize from file
    rag.initialize_from_file(file_path)
    
    # Example questions
    questions = [
        "What is LangChain?",
        "What are the key features of LangChain?",
        "What are some use cases for LangChain?",
        "What is RAG in LangChain?"
    ]
    
    print("Asking example questions...")
    print("=" * 60)
    
    for question in questions:
        print(f"\nQ: {question}")
        result = rag.ask_question(question)
        print(f"A: {result['answer']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
