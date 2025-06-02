import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from rich.console import Console
# import os # Not used
# import subprocess # Not used
from tonic_ollama_client import (
    TonicOllamaClient,
    create_client,
    OllamaServerNotRunningError,
    ResponseError,
)
from ollama import AsyncClient as OllamaAsyncClient # Keep for type hinting if needed, but not for isinstance on mock

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
            # .list and .pull are not directly used by TonicOllamaClient's core logic anymore
            # mock_ollama_instance.list = AsyncMock() 
            # mock_ollama_instance.pull = AsyncMock()
            
            client = TonicOllamaClient(debug=False)
            # Store the mock for assertion if needed, though get_async_client returns it
            yield client, mock_ollama_instance

    def test_initialization_defaults(self):
        client = TonicOllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.max_server_startup_attempts == 3  # Changed attribute
        assert client.debug is False
        assert isinstance(client.console, Console)
        assert client.conversations == {}

    def test_initialization_custom(self):
        custom_console = Console()
        client = TonicOllamaClient(
            base_url="http://test-url:1234",
            max_server_startup_attempts=5,  # Changed parameter and attribute
            debug=True,
            console=custom_console
        )
        assert client.base_url == "http://test-url:1234"
        assert client.max_server_startup_attempts == 5  # Changed attribute
        assert client.debug is True
        assert client.console == custom_console

    def test_get_async_client_creates_once(self, client_instance):
        client, mock_ollama_instance_from_fixture = client_instance
        first_async_client = client.get_async_client()
        # Assert it's the mock instance we expect from the patch
        assert first_async_client is mock_ollama_instance_from_fixture
        second_async_client = client.get_async_client()
        assert first_async_client is second_async_client

    @patch('socket.create_connection', return_value=MagicMock())
    def test_is_ollama_server_running_sync_true(self, mock_socket_conn, client_instance): # Renamed test
        client, _ = client_instance
        assert client._is_ollama_server_running_sync() is True # Use _sync version
        mock_socket_conn.assert_called_once()

    @patch('socket.create_connection', side_effect=ConnectionRefusedError)
    def test_is_ollama_server_running_sync_false(self, mock_socket_conn, client_instance): # Renamed test
        client, _ = client_instance
        assert client._is_ollama_server_running_sync() is False # Use _sync version
        mock_socket_conn.assert_called_once()

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running_sync', return_value=True) # Patch _sync version
    async def test_ensure_server_ready_server_responsive(self, mock_is_server_running_sync, client_instance): # Renamed and refactored
        client, _ = client_instance # mock_ollama not needed here
        
        await client.ensure_server_ready() # No model_name argument
        mock_is_server_running_sync.assert_called_once()
        # Assertions for mock_ollama.list, chat, pull are removed as ensure_server_ready doesn't do this

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running_sync', side_effect=[False, False, True]) # Patch _sync version
    async def test_ensure_server_ready_becomes_responsive_after_retries(self, mock_is_server_running_sync, client_instance): # New test
        client, _ = client_instance
        client.max_server_startup_attempts = 3
        
        await client.ensure_server_ready()
        assert mock_is_server_running_sync.call_count == 3


    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running_sync', return_value=False) # Patch _sync version
    async def test_ensure_server_ready_server_not_running_raises_error(self, mock_is_server_running_sync, client_instance): # Renamed and refactored
        client, _ = client_instance
        client.max_server_startup_attempts = 2 # For faster test
        with pytest.raises(OllamaServerNotRunningError):
            await client.ensure_server_ready() # No model_name argument
        assert mock_is_server_running_sync.call_count == 2

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running_sync', return_value=False) # Patch _sync version
    async def test_chat_server_not_running_via_ensure_ready(self, mock_is_server_running_sync, client_instance): # Renamed and refactored
        client, _ = client_instance # mock_ollama not needed for this specific failure path
        client.max_server_startup_attempts = 1 # Ensure it fails quickly
        
        # Reset the mock to clear any previous calls
        mock_is_server_running_sync.reset_mock()
        
        with pytest.raises(OllamaServerNotRunningError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="Hello")
        
        # Verify it was called at least once - we don't care about exact count
        # due to potential retries from both ensure_server_ready and API_RETRY_CONFIG
        assert mock_is_server_running_sync.call_count > 0, "Server check should be called at least once"

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running_sync', return_value=False) # Patch _sync version
    async def test_generate_embedding_server_not_running_via_ensure_ready(self, mock_is_server_running_sync, client_instance): # Renamed and refactored
        client, _ = client_instance # mock_ollama not needed for this specific failure path
        client.max_server_startup_attempts = 1 # Ensure it fails quickly
        
        # Reset the mock to clear any previous calls
        mock_is_server_running_sync.reset_mock()
        
        with pytest.raises(OllamaServerNotRunningError):
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="Embedding test")
        
        # Verify it was called at least once - we don't care about exact count
        # due to potential retries from both ensure_server_ready and API_RETRY_CONFIG
        assert mock_is_server_running_sync.call_count > 0, "Server check should be called at least once"

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