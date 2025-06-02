import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from rich.console import Console
import os
import subprocess
from tonic_ollama_client import (
    TonicOllamaClient,
    create_client,
    ModelNotReadyError,
    OllamaServerNotRunningError,
    ResponseError,
    fancy_print,
)
from ollama import AsyncClient as OllamaAsyncClient

APPROVED_MODELS = ["llama3.1:latest", "phi4:latest", "qwen2:7b"]
DEFAULT_TEST_MODEL = "llama3.1:latest"

class TestTonicOllamaClientMethods:
    """Tests for TonicOllamaClient class methods."""

    @pytest_asyncio.fixture
    async def client_instance(self):
        """Provides a TonicOllamaClient instance with mocked AsyncClient."""
        with patch('tonic_ollama_client.AsyncClient') as MockedOllamaAsyncClient:
            mock_ollama_instance = MockedOllamaAsyncClient.return_value
            mock_ollama_instance.chat = AsyncMock()
            mock_ollama_instance.embeddings = AsyncMock()
            mock_ollama_instance.list = AsyncMock()
            mock_ollama_instance.pull = AsyncMock()
            
            client = TonicOllamaClient(debug=False)
            yield client, mock_ollama_instance

    def test_initialization_defaults(self):
        client = TonicOllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.max_readiness_attempts == 3
        assert client.debug is False
        assert isinstance(client.console, Console)
        assert client.async_client is None

    def test_initialization_custom(self):
        custom_console = Console()
        client = TonicOllamaClient(
            base_url="http://test-url:1234",
            max_readiness_attempts=5,
            debug=True,
            console=custom_console
        )
        assert client.base_url == "http://test-url:1234"
        assert client.max_readiness_attempts == 5
        assert client.debug is True
        assert client.console == custom_console

    def test_get_async_client_creates_once(self, client_instance):
        client, _ = client_instance
        first_async_client = client.get_async_client()
        assert isinstance(first_async_client, OllamaAsyncClient)
        second_async_client = client.get_async_client()
        assert first_async_client is second_async_client

    @patch('socket.create_connection', return_value=MagicMock())
    def test_is_ollama_server_running_true(self, mock_socket_conn, client_instance):
        client, _ = client_instance
        assert client._is_ollama_server_running() is True
        mock_socket_conn.assert_called_once()

    @patch('socket.create_connection', side_effect=ConnectionRefusedError)
    def test_is_ollama_server_running_false(self, mock_socket_conn, client_instance):
        client, _ = client_instance
        assert client._is_ollama_server_running() is False
        mock_socket_conn.assert_called_once()

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running', return_value=True)
    async def test_check_model_ready_server_running_model_exists_responsive(self, mock_is_server_running, client_instance):
        client, mock_ollama = client_instance
        
        mock_ollama.list.return_value = {"models": [{"name": DEFAULT_TEST_MODEL}]}
        mock_ollama.chat.return_value = {"message": {"content": "READY"}}
        
        await client.check_model_ready(DEFAULT_TEST_MODEL)
        mock_is_server_running.assert_called_once()
        mock_ollama.list.assert_called_once()
        mock_ollama.chat.assert_called_once()
        mock_ollama.pull.assert_not_called()

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running', return_value=True)
    async def test_check_model_ready_needs_pull(self, mock_is_server_running, client_instance):
        client, mock_ollama = client_instance
        
        mock_ollama.list.return_value = {"models": []} # Model not found
        mock_ollama.pull.return_value = {"status": "success"}
        mock_ollama.chat.return_value = {"message": {"content": "READY"}}
        
        await client.check_model_ready(DEFAULT_TEST_MODEL)
        mock_ollama.pull.assert_called_once_with(model=DEFAULT_TEST_MODEL, stream=False)
        mock_ollama.chat.assert_called_once()

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running', return_value=False)
    async def test_check_model_ready_server_not_running(self, mock_is_server_running, client_instance):
        client, _ = client_instance
        with pytest.raises(OllamaServerNotRunningError):
            await client.check_model_ready(DEFAULT_TEST_MODEL)
        mock_is_server_running.assert_called_once()

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running', return_value=True)
    async def test_chat_server_not_running(self, mock_is_server_running, client_instance):
        client, _ = client_instance
        mock_is_server_running.return_value = False
        with pytest.raises(OllamaServerNotRunningError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="Hello")
        mock_is_server_running.assert_called_once()

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running', return_value=True)
    async def test_generate_embedding_server_not_running(self, mock_is_server_running, client_instance):
        client, _ = client_instance
        mock_is_server_running.return_value = False
        with pytest.raises(OllamaServerNotRunningError):
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="Embedding test")
        mock_is_server_running.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_management(self, client_instance):
        client, _ = client_instance
        conv_id = await client.create_conversation("test-conv")
        assert "test-conv" in client.list_conversations()
        
        client.conversations[conv_id].append({"role": "user", "content": "message1"})
        messages = client.get_conversation(conv_id)
        assert len(messages) == 1
        
        client.clear_conversation(conv_id)
        assert len(client.get_conversation(conv_id)) == 0
        
        client.delete_conversation(conv_id)
        assert "test-conv" not in client.list_conversations()
        with pytest.raises(ValueError):
            client.get_conversation(conv_id)

def test_create_client_default():
    client = create_client()
    assert isinstance(client, TonicOllamaClient)
    assert client.debug is False

def test_create_client_custom():
    console = Console()
    client = create_client(base_url="http://custom:1111", debug=True, console=console)
    assert client.base_url == "http://custom:1111"
    assert client.debug is True
    assert client.console is console
