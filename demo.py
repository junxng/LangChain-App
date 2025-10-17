#!/usr/bin/env python3
"""
Demonstration script for the RAG Pipeline

This script demonstrates the capabilities of the RAG pipeline
with or without an API key. Without an API key, it shows the
structure and usage. With an API key, it runs actual queries.
"""

import os
from rag_app import RAGPipeline


def demo_without_api():
    """Demonstrate the pipeline structure without API key"""
    print("=" * 70)
    print("LangChain RAG Pipeline - Demo Mode (No API Key)")
    print("=" * 70)
    print()
    print("This is a demonstration of how the RAG pipeline works.")
    print()
    print("1. Document Loading:")
    print("   - Load text files using LangChain's TextLoader")
    print("   - Supports UTF-8 encoded text files")
    print()
    print("2. Text Splitting:")
    print("   - Documents are split into chunks (default: 1000 chars)")
    print("   - Overlap between chunks (default: 200 chars) for context")
    print("   - Uses RecursiveCharacterTextSplitter for intelligent splitting")
    print()
    print("3. Embedding & Vector Store:")
    print("   - Each chunk is embedded using OpenAI's embedding model")
    print("   - Embeddings are stored in FAISS for fast similarity search")
    print()
    print("4. Question Answering:")
    print("   - Questions are embedded using the same model")
    print("   - Top-k most relevant chunks are retrieved (default: k=3)")
    print("   - Context + question is sent to GPT-3.5-turbo")
    print("   - Model generates an answer based on the context")
    print()
    print("=" * 70)
    print()
    print("To run with actual API, set your OpenAI API key:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    print()
    print("Then run:")
    print("  python demo.py")
    print("  or")
    print("  python cli.py sample_data.txt")
    print()


def demo_with_api():
    """Demonstrate the pipeline with actual API"""
    print("=" * 70)
    print("LangChain RAG Pipeline - Live Demo")
    print("=" * 70)
    print()
    
    # Check if sample file exists
    if not os.path.exists("sample_data.txt"):
        print("Error: sample_data.txt not found!")
        return
    
    try:
        # Initialize pipeline
        print("Initializing RAG Pipeline...")
        rag = RAGPipeline()
        
        # Load and process document
        print(f"Loading document: sample_data.txt")
        rag.initialize_from_file("sample_data.txt")
        
        # Example questions
        questions = [
            "What is LangChain?",
            "What are the key features of LangChain?",
            "What is RAG in the context of LangChain?",
        ]
        
        print("\n" + "=" * 70)
        print("Example Questions and Answers")
        print("=" * 70)
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. Question: {question}")
            result = rag.ask_question(question)
            print(f"   Answer: {result['answer']}")
            print(f"   (Based on {len(result['source_documents'])} source chunks)")
            print("-" * 70)
        
        print("\n✓ Demo completed successfully!")
        print("\nTo ask your own questions, use the interactive CLI:")
        print("  python cli.py sample_data.txt")
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        print("\nPlease check:")
        print("  1. OPENAI_API_KEY is set correctly")
        print("  2. You have credits in your OpenAI account")
        print("  3. sample_data.txt exists")


def main():
    """Main demo function"""
    if os.getenv("OPENAI_API_KEY"):
        demo_with_api()
    else:
        demo_without_api()


if __name__ == "__main__":
    main()
