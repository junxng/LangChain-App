# Quick Start Guide

This guide will help you get started with the LangChain RAG Pipeline quickly.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/junxng/LangChain-App.git
   cd LangChain-App
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   Or create a `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

## Usage Examples

### 1. Demo Mode (No API Key Required)
```bash
python demo.py
```
Shows how the RAG pipeline works without making actual API calls.

### 2. Interactive Mode
```bash
python cli.py sample_data.txt
```
Start an interactive session where you can ask multiple questions.

Example interaction:
```
Your question: What is LangChain?
Thinking...
Answer: LangChain is a framework for developing applications powered by language models...

Your question: What are the key features?
Thinking...
Answer: The key features include Chains, Agents, Memory, Document Loaders...
```

### 3. Single Question Mode
```bash
python cli.py sample_data.txt -q "What is RAG?"
```
Ask a single question and get an immediate answer.

### 4. Use Your Own Documents
```bash
python cli.py /path/to/your/document.txt
```
Ask questions about any text file.

### 5. Programmatic Usage
```python
from rag_app import RAGPipeline

# Initialize
rag = RAGPipeline(openai_api_key="your-key")

# Load document
rag.initialize_from_file("sample_data.txt")

# Ask questions
result = rag.ask_question("What is LangChain?")
print(result['answer'])
```

## Common Options

### Custom Chunk Size
For better results with different document types:
```bash
# Smaller chunks for factual data
python cli.py document.txt --chunk-size 500 --chunk-overlap 100

# Larger chunks for narrative content
python cli.py document.txt --chunk-size 2000 --chunk-overlap 200
```

### Provide API Key Directly
```bash
python cli.py sample_data.txt --api-key YOUR_API_KEY
```

## Troubleshooting

### "No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### "OpenAI API key not provided"
Make sure your API key is set:
```bash
export OPENAI_API_KEY='your-api-key'
```
Or use the `--api-key` parameter.

### "Rate limit exceeded"
Your OpenAI account may have hit rate limits. Wait a moment and try again, or upgrade your OpenAI plan.

### File encoding issues
Ensure your text files are UTF-8 encoded:
```bash
file -i your-file.txt  # Check encoding
iconv -f ISO-8859-1 -t UTF-8 your-file.txt > new-file.txt  # Convert if needed
```

## Testing

Run the test suite:
```bash
python -m unittest test_rag_app -v
```

## Next Steps

- Read the full [README.md](README.md) for detailed information
- Try different chunk sizes to optimize for your documents
- Experiment with different types of questions
- Use your own documents to build a custom knowledge base

## Getting Help

- Check the [README.md](README.md) for full documentation
- Review example questions in `demo.py`
- See the API documentation in `rag_app.py`

## Cost Considerations

The RAG pipeline uses OpenAI's API, which incurs costs:
- **Embeddings**: ~$0.0001 per 1000 tokens
- **GPT-3.5-turbo**: ~$0.0015 per 1000 tokens

A typical question-answer cycle on a small document costs less than $0.01.

Monitor your usage at: https://platform.openai.com/usage
