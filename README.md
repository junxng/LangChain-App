# LangChain RAG Application

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain that allows users to ask questions about the contents of text files.

## Features

- 📄 Load and process text documents
- 🔍 Intelligent document retrieval using vector embeddings
- 💬 Question-answering using OpenAI's GPT models
- 🖥️ Interactive CLI interface
- ⚙️ Configurable chunk size and overlap for optimal retrieval

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language generation. Instead of relying solely on the knowledge embedded in a language model, RAG:

1. **Retrieves** relevant documents from a knowledge base
2. **Augments** the prompt with this retrieved context
3. **Generates** an answer based on both the retrieved information and the model's knowledge

This approach provides more accurate, contextual, and verifiable answers.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/junxng/LangChain-App.git
cd LangChain-App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Or export it directly:
export OPENAI_API_KEY='your-api-key-here'
```

Get your API key from: https://platform.openai.com/api-keys

## Usage

### Interactive Mode

Run the CLI in interactive mode to ask multiple questions:

```bash
python cli.py sample_data.txt
```

This will start an interactive session where you can ask questions about the document.

### Single Question Mode

Ask a single question and exit:

```bash
python cli.py sample_data.txt -q "What is LangChain?"
```

### Using Your Own Documents

You can use any text file:

```bash
python cli.py path/to/your/document.txt
```

### Advanced Options

```bash
# Custom chunk size and overlap
python cli.py sample_data.txt --chunk-size 500 --chunk-overlap 100

# Provide API key directly (without environment variable)
python cli.py sample_data.txt --api-key YOUR_API_KEY
```

### Programmatic Usage

You can also use the RAG pipeline in your own Python code:

```python
from rag_app import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline(openai_api_key="your-api-key-here")

# Load and process a document
rag.initialize_from_file("sample_data.txt")

# Ask questions
result = rag.ask_question("What is LangChain?")
print(f"Q: {result['question']}")
print(f"A: {result['answer']}")

# The result also contains source documents
print(f"Sources: {len(result['source_documents'])} document(s)")
```

## How It Works

1. **Document Loading**: The application loads your text file using LangChain's document loaders
2. **Text Splitting**: The document is split into smaller chunks for better retrieval (default: 1000 characters with 200 character overlap)
3. **Embedding Creation**: Each chunk is converted into a vector embedding using OpenAI's embedding model
4. **Vector Store**: Embeddings are stored in a FAISS vector database for efficient retrieval
5. **Question Processing**: When you ask a question:
   - The question is embedded using the same model
   - Similar document chunks are retrieved from the vector store
   - The retrieved context is provided to GPT-3.5-turbo along with your question
   - The model generates an answer based on the context

## Project Structure

```
LangChain-App/
├── rag_app.py          # Main RAG pipeline implementation
├── cli.py              # Interactive command-line interface
├── requirements.txt    # Python dependencies
├── sample_data.txt     # Sample document about LangChain
├── .env.example        # Environment variable template
└── README.md           # This file
```

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`

## Example Questions

Try asking these questions about the sample data:

- "What is LangChain?"
- "What are the key features of LangChain?"
- "What are some use cases for LangChain?"
- "What is RAG in LangChain?"
- "How does LangChain help with document loading?"

## Customization

### Chunk Size

Adjust the chunk size based on your documents:
- Larger chunks (1000-2000): Better for documents with long, coherent sections
- Smaller chunks (200-500): Better for documents with distinct, short pieces of information

### Retrieval Parameters

In `rag_app.py`, you can modify the retrieval parameters:
```python
retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
```
Change `k` to retrieve more or fewer relevant chunks (default: 3).

### LLM Model

Change the model in `rag_app.py`:
```python
self.llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4",  # or "gpt-3.5-turbo"
    openai_api_key=self.api_key
)
```

## Troubleshooting

### API Key Issues
- Make sure your OpenAI API key is valid and has sufficient credits
- Ensure the key is properly set in your environment or passed as a parameter

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using Python 3.8 or higher

### Memory Issues
- For very large documents, consider increasing chunk size or processing in batches
- The in-memory vector store may not be suitable for very large document collections

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Uses [OpenAI's GPT models](https://openai.com/)
- Vector storage powered by [FAISS](https://github.com/facebookresearch/faiss)