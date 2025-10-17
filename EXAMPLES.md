# Usage Examples

This document provides comprehensive examples of using the LangChain RAG Pipeline.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Advanced Usage](#advanced-usage)
- [Integration Examples](#integration-examples)
- [Best Practices](#best-practices)

## Basic Usage

### Example 1: Simple Question Answering

```python
from rag_app import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline(openai_api_key="your-api-key")

# Load a document
rag.initialize_from_file("sample_data.txt")

# Ask a question
result = rag.ask_question("What is LangChain?")
print(f"Answer: {result['answer']}")
```

### Example 2: Multiple Questions

```python
from rag_app import RAGPipeline

rag = RAGPipeline(openai_api_key="your-api-key")
rag.initialize_from_file("sample_data.txt")

questions = [
    "What is LangChain?",
    "What are the key features?",
    "What is RAG?",
]

for question in questions:
    result = rag.ask_question(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
```

### Example 3: Using Environment Variables

```python
import os
from dotenv import load_dotenv
from rag_app import RAGPipeline

# Load .env file
load_dotenv()

# API key is automatically read from environment
rag = RAGPipeline()
rag.initialize_from_file("sample_data.txt")

result = rag.ask_question("Tell me about LangChain")
print(result['answer'])
```

## Advanced Usage

### Example 4: Custom Chunk Settings

```python
from rag_app import RAGPipeline

rag = RAGPipeline(openai_api_key="your-api-key")

# Use smaller chunks for more precise retrieval
rag.initialize_from_file(
    "technical_doc.txt",
    chunk_size=500,
    chunk_overlap=100
)

result = rag.ask_question("What is the API endpoint?")
print(result['answer'])
```

### Example 5: Step-by-Step Pipeline

```python
from rag_app import RAGPipeline

# Initialize
rag = RAGPipeline(openai_api_key="your-api-key")

# Load document
documents = rag.load_document("my_document.txt")
print(f"Loaded {len(documents)} document(s)")

# Split into chunks
chunks = rag.split_documents(documents, chunk_size=800, chunk_overlap=150)
print(f"Created {len(chunks)} chunk(s)")

# Create vector store
rag.create_vectorstore(chunks)

# Setup QA chain
rag.setup_qa_chain()

# Ask questions
result = rag.ask_question("What is the main topic?")
print(result['answer'])
```

### Example 6: Accessing Source Documents

```python
from rag_app import RAGPipeline

rag = RAGPipeline(openai_api_key="your-api-key")
rag.initialize_from_file("sample_data.txt")

result = rag.ask_question("What are LangChain's use cases?")

print(f"Answer: {result['answer']}\n")
print(f"Based on {len(result['source_documents'])} sources:")

for i, doc in enumerate(result['source_documents'], 1):
    print(f"\nSource {i}:")
    print(f"  Content: {doc.page_content[:100]}...")
```

## Integration Examples

### Example 7: Web API Integration (Flask)

```python
from flask import Flask, request, jsonify
from rag_app import RAGPipeline

app = Flask(__name__)

# Initialize RAG pipeline once
rag = RAGPipeline(openai_api_key="your-api-key")
rag.initialize_from_file("knowledge_base.txt")

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        result = rag.ask_question(question)
        return jsonify({
            'question': question,
            'answer': result['answer'],
            'num_sources': len(result['source_documents'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Example 8: Batch Processing

```python
from rag_app import RAGPipeline
import json

def batch_process_questions(document_path, questions_file, output_file):
    """Process multiple questions and save results"""
    
    # Load questions
    with open(questions_file, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Initialize RAG
    rag = RAGPipeline(openai_api_key="your-api-key")
    rag.initialize_from_file(document_path)
    
    # Process questions
    results = []
    for i, question in enumerate(questions, 1):
        print(f"Processing {i}/{len(questions)}: {question}")
        result = rag.ask_question(question)
        results.append({
            'question': question,
            'answer': result['answer']
        })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

# Usage
batch_process_questions(
    "knowledge_base.txt",
    "questions.txt",
    "answers.json"
)
```

### Example 9: Multiple Document Sources

```python
from rag_app import RAGPipeline

def load_multiple_documents(file_paths):
    """Load and combine multiple documents"""
    rag = RAGPipeline(openai_api_key="your-api-key")
    
    all_chunks = []
    for file_path in file_paths:
        print(f"Loading {file_path}...")
        docs = rag.load_document(file_path)
        chunks = rag.split_documents(docs)
        all_chunks.extend(chunks)
    
    print(f"Total chunks: {len(all_chunks)}")
    rag.create_vectorstore(all_chunks)
    rag.setup_qa_chain()
    
    return rag

# Usage
files = [
    "doc1.txt",
    "doc2.txt",
    "doc3.txt"
]

rag = load_multiple_documents(files)
result = rag.ask_question("What is the common theme?")
print(result['answer'])
```

## Best Practices

### Example 10: Error Handling

```python
from rag_app import RAGPipeline
import os

def safe_rag_query(document_path, question):
    """Query with comprehensive error handling"""
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        return {
            'error': 'OpenAI API key not set',
            'suggestion': 'Set OPENAI_API_KEY environment variable'
        }
    
    # Check document exists
    if not os.path.exists(document_path):
        return {
            'error': f'Document not found: {document_path}',
            'suggestion': 'Check the file path'
        }
    
    try:
        # Initialize and query
        rag = RAGPipeline()
        rag.initialize_from_file(document_path)
        result = rag.ask_question(question)
        
        return {
            'success': True,
            'answer': result['answer'],
            'sources': len(result['source_documents'])
        }
        
    except ValueError as e:
        return {
            'error': 'Configuration error',
            'details': str(e)
        }
    except Exception as e:
        return {
            'error': 'Unexpected error',
            'details': str(e)
        }

# Usage
result = safe_rag_query("sample_data.txt", "What is LangChain?")
if 'error' in result:
    print(f"Error: {result['error']}")
    if 'suggestion' in result:
        print(f"Suggestion: {result['suggestion']}")
else:
    print(f"Answer: {result['answer']}")
```

### Example 11: Performance Monitoring

```python
from rag_app import RAGPipeline
import time

def timed_query(rag, question):
    """Measure query performance"""
    start_time = time.time()
    result = rag.ask_question(question)
    end_time = time.time()
    
    return {
        'question': question,
        'answer': result['answer'],
        'execution_time': end_time - start_time,
        'num_sources': len(result['source_documents'])
    }

# Usage
rag = RAGPipeline(openai_api_key="your-api-key")
rag.initialize_from_file("sample_data.txt")

questions = [
    "What is LangChain?",
    "What are its features?",
    "What is RAG?"
]

for question in questions:
    result = timed_query(rag, question)
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Time: {result['execution_time']:.2f}s")
    print(f"Sources: {result['num_sources']}")
```

### Example 12: Custom Prompt Template

```python
from rag_app import RAGPipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize RAG
rag = RAGPipeline(openai_api_key="your-api-key")
documents = rag.load_document("sample_data.txt")
chunks = rag.split_documents(documents)
rag.create_vectorstore(chunks)

# Custom prompt for technical documentation
tech_prompt = PromptTemplate(
    template="""You are a technical documentation assistant. 
    Use the following context to provide a precise, technical answer.
    If the answer is not in the context, say "This information is not available in the documentation."
    
    Context: {context}
    
    Question: {question}
    
    Technical Answer:""",
    input_variables=["context", "question"]
)

# Setup QA with custom prompt
rag.qa_chain = RetrievalQA.from_chain_type(
    llm=rag.llm,
    chain_type="stuff",
    retriever=rag.vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": tech_prompt}
)

# Ask question
result = rag.ask_question("What are the technical requirements?")
print(result['answer'])
```

## Tips for Best Results

1. **Chunk Size**: 
   - Technical docs: 500-800 characters
   - Narrative content: 1000-1500 characters
   - Short form: 300-500 characters

2. **Questions**:
   - Be specific and clear
   - Use keywords from the document
   - Break complex questions into smaller ones

3. **Document Preparation**:
   - Use UTF-8 encoding
   - Remove unnecessary formatting
   - Keep related information together

4. **Performance**:
   - Reuse the same RAG instance for multiple queries
   - Cache embeddings when possible
   - Monitor API usage and costs
