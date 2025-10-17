#!/usr/bin/env python3
"""
Interactive CLI for the RAG Question-Answering System

This script provides an interactive command-line interface for asking
questions about documents using the RAG pipeline.
"""

import os
import sys
import argparse
from rag_app import RAGPipeline


def interactive_mode(rag: RAGPipeline):
    """
    Run the RAG pipeline in interactive mode.
    
    Args:
        rag: Initialized RAGPipeline instance.
    """
    print("\n" + "=" * 60)
    print("Interactive Question-Answering Mode")
    print("=" * 60)
    print("\nType your questions below. Type 'quit', 'exit', or press Ctrl+C to exit.\n")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if not question:
                print("Please enter a question.")
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            print("\nThinking...")
            result = rag.ask_question(question)
            print(f"\nAnswer: {result['answer']}")
            
            if result.get('source_documents'):
                print(f"\n(Based on {len(result['source_documents'])} source document(s))")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


def single_question_mode(rag: RAGPipeline, question: str):
    """
    Ask a single question and exit.
    
    Args:
        rag: Initialized RAGPipeline instance.
        question: The question to ask.
    """
    print(f"\nQuestion: {question}")
    print("\nThinking...")
    
    result = rag.ask_question(question)
    print(f"\nAnswer: {result['answer']}\n")


def main():
    """
    Main function for the CLI application.
    """
    parser = argparse.ArgumentParser(
        description="RAG-based Question Answering System using LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with sample data
  python cli.py sample_data.txt

  # Ask a single question
  python cli.py sample_data.txt -q "What is LangChain?"

  # Use custom chunk size
  python cli.py my_document.txt --chunk-size 500 --chunk-overlap 100

  # Provide API key directly
  python cli.py sample_data.txt --api-key YOUR_API_KEY
        """
    )
    
    parser.add_argument(
        "file",
        help="Path to the text file to load"
    )
    
    parser.add_argument(
        "-q", "--question",
        help="Ask a single question and exit (non-interactive mode)"
    )
    
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (alternatively set OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks (default: 1000)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    # Check for API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided.")
        print("Please either:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Use --api-key parameter")
        print("\nExample: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    try:
        # Initialize the RAG pipeline
        print("Initializing RAG Pipeline...")
        rag = RAGPipeline(openai_api_key=api_key)
        
        # Load and process the document
        print(f"Loading document: {args.file}")
        rag.initialize_from_file(
            args.file,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Run in appropriate mode
        if args.question:
            single_question_mode(rag, args.question)
        else:
            interactive_mode(rag)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
