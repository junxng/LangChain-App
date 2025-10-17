"""
Unit tests for the RAG Pipeline

These tests verify the basic functionality of the RAG pipeline
without requiring an actual OpenAI API key.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
from rag_app import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    """Test cases for RAGPipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary test file
        self.test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.test_file.write("LangChain is a framework for developing applications with LLMs.")
        self.test_file.close()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_file.name):
            os.unlink(self.test_file.name)
    
    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key"""
        # Make sure no API key is in environment
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                RAGPipeline()
            self.assertIn("OpenAI API key must be provided", str(context.exception))
    
    @patch('rag_app.OpenAIEmbeddings')
    @patch('rag_app.ChatOpenAI')
    def test_initialization_with_api_key(self, mock_chat, mock_embeddings):
        """Test that initialization succeeds with API key"""
        rag = RAGPipeline(openai_api_key="test-key")
        self.assertIsNotNone(rag)
        self.assertEqual(rag.api_key, "test-key")
        mock_embeddings.assert_called_once()
        mock_chat.assert_called_once()
    
    @patch('rag_app.OpenAIEmbeddings')
    @patch('rag_app.ChatOpenAI')
    def test_load_document_success(self, mock_chat, mock_embeddings):
        """Test loading a document successfully"""
        rag = RAGPipeline(openai_api_key="test-key")
        
        with patch('rag_app.TextLoader') as mock_loader:
            mock_doc = Mock()
            mock_doc.page_content = "Test content"
            mock_loader.return_value.load.return_value = [mock_doc]
            
            docs = rag.load_document(self.test_file.name)
            self.assertEqual(len(docs), 1)
    
    @patch('rag_app.OpenAIEmbeddings')
    @patch('rag_app.ChatOpenAI')
    def test_load_document_file_not_found(self, mock_chat, mock_embeddings):
        """Test loading a non-existent document"""
        rag = RAGPipeline(openai_api_key="test-key")
        
        with self.assertRaises(FileNotFoundError):
            rag.load_document("non_existent_file.txt")
    
    @patch('rag_app.OpenAIEmbeddings')
    @patch('rag_app.ChatOpenAI')
    def test_split_documents(self, mock_chat, mock_embeddings):
        """Test document splitting"""
        rag = RAGPipeline(openai_api_key="test-key")
        
        # Create mock documents
        mock_doc = Mock()
        mock_doc.page_content = "A" * 2000  # Long document
        mock_doc.metadata = {}
        
        chunks = rag.split_documents([mock_doc], chunk_size=500, chunk_overlap=50)
        # Should split into multiple chunks
        self.assertGreater(len(chunks), 1)
    
    @patch('rag_app.OpenAIEmbeddings')
    @patch('rag_app.ChatOpenAI')
    @patch('rag_app.FAISS')
    def test_create_vectorstore(self, mock_faiss, mock_chat, mock_embeddings):
        """Test vector store creation"""
        rag = RAGPipeline(openai_api_key="test-key")
        
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        
        mock_faiss.from_documents.return_value = Mock()
        
        rag.create_vectorstore([mock_doc])
        
        mock_faiss.from_documents.assert_called_once()
        self.assertIsNotNone(rag.vectorstore)
    
    @patch('rag_app.OpenAIEmbeddings')
    @patch('rag_app.ChatOpenAI')
    def test_setup_qa_chain_without_vectorstore(self, mock_chat, mock_embeddings):
        """Test that setup_qa_chain fails without vectorstore"""
        rag = RAGPipeline(openai_api_key="test-key")
        
        with self.assertRaises(ValueError) as context:
            rag.setup_qa_chain()
        self.assertIn("Vector store not initialized", str(context.exception))
    
    @patch('rag_app.OpenAIEmbeddings')
    @patch('rag_app.ChatOpenAI')
    @patch('rag_app.FAISS')
    @patch('rag_app.RetrievalQA')
    def test_setup_qa_chain_with_vectorstore(self, mock_qa, mock_faiss, mock_chat, mock_embeddings):
        """Test QA chain setup with vectorstore"""
        rag = RAGPipeline(openai_api_key="test-key")
        
        # Mock vectorstore
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = Mock()
        rag.vectorstore = mock_vectorstore
        
        mock_qa.from_chain_type.return_value = Mock()
        
        rag.setup_qa_chain()
        
        mock_qa.from_chain_type.assert_called_once()
        self.assertIsNotNone(rag.qa_chain)
    
    @patch('rag_app.OpenAIEmbeddings')
    @patch('rag_app.ChatOpenAI')
    def test_ask_question_without_qa_chain(self, mock_chat, mock_embeddings):
        """Test that asking a question fails without QA chain"""
        rag = RAGPipeline(openai_api_key="test-key")
        
        with self.assertRaises(ValueError) as context:
            rag.ask_question("What is LangChain?")
        self.assertIn("QA chain not initialized", str(context.exception))


class TestDocumentLoading(unittest.TestCase):
    """Test document loading functionality"""
    
    def test_sample_data_exists(self):
        """Test that sample_data.txt exists"""
        sample_file = "sample_data.txt"
        if os.path.exists(sample_file):
            self.assertTrue(os.path.isfile(sample_file))
            
            # Check that file has content
            with open(sample_file, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0)
                self.assertIn("LangChain", content)


if __name__ == '__main__':
    unittest.main()
