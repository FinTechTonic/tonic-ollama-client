import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from tonic_ollama_client import TonicOllamaClient, ResponseError, OllamaServerNotRunningError # Replaced ModelNotReadyError
import asyncio

APPROVED_MODELS = ["llama3.1:latest", "phi4:latest", "qwen3:8b", "mistral:latest"]
DEFAULT_TEST_MODEL = "llama3.1:latest"

class TestErrorHandling:
    """Test comprehensive error handling scenarios."""
    
    @pytest_asyncio.fixture
    async def mock_client_with_errors(self):
        """Fixture for a client with mocked error scenarios."""
        with patch('tonic_ollama_client.AsyncClient') as MockedAsyncClient:
            mock_instance = MockedAsyncClient.return_value
            mock_instance.chat = AsyncMock()
            mock_instance.embeddings = AsyncMock()
            
            # Patch _is_ollama_server_running_sync for tests that need to control server status
            with patch.object(TonicOllamaClient, '_is_ollama_server_running_sync', return_value=True) as mock_server_check:
                client = TonicOllamaClient(debug=True)
                # Yield the mock_server_check as well if tests need to manipulate it
                yield client, mock_instance, mock_server_check # Corrected mock_ollama to mock_instance
    
    @pytest.mark.asyncio
    async def test_chat_connection_error(self, mock_client_with_errors):
        """Test chat with connection error."""
        client, mock_ollama, _ = mock_client_with_errors # Unpack mock_server_check
        
        # Simulate server check passes, but chat fails
        mock_ollama.chat.side_effect = ConnectionError("Connection refused")
        
        with pytest.raises(ConnectionError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="test")
    
    @pytest.mark.asyncio
    async def test_chat_timeout_error(self, mock_client_with_errors):
        """Test chat with timeout error."""
        client, mock_ollama, _ = mock_client_with_errors
        
        mock_ollama.chat.side_effect = TimeoutError("Request timed out")
        
        with pytest.raises(TimeoutError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="test")
    
    @pytest.mark.asyncio
    async def test_chat_response_error_various_codes(self, mock_client_with_errors):
        """Test chat with various HTTP response errors."""
        client, mock_ollama, _ = mock_client_with_errors
        
        error_scenarios = [
            (404, "Model not found"),
            (500, "Internal server error"),
            (503, "Service unavailable"),
            (401, "Unauthorized"),
        ]
        
        for status_code, error_message in error_scenarios:
            mock_ollama.chat.side_effect = ResponseError(error_message, status_code)
            
            with pytest.raises(ResponseError) as exc_info:
                await client.chat(model=DEFAULT_TEST_MODEL, message="test")
            
            assert exc_info.value.status_code == status_code
            assert error_message in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_embedding_connection_error(self, mock_client_with_errors):
        """Test embedding generation with connection error."""
        client, mock_ollama, _ = mock_client_with_errors
        
        mock_ollama.embeddings.side_effect = ConnectionError("Network unreachable")
        
        with pytest.raises(ConnectionError):
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="test")
    
    @pytest.mark.asyncio
    async def test_embedding_response_error(self, mock_client_with_errors):
        """Test embedding generation with response error."""
        client, mock_ollama, _ = mock_client_with_errors
        
        mock_ollama.embeddings.side_effect = ResponseError("Embedding failed", 422)
        
        with pytest.raises(ResponseError) as exc_info:
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="test")
        
        assert exc_info.value.status_code == 422
    
    @pytest.mark.asyncio
    async def test_conversation_errors(self):
        """Test conversation management error scenarios."""
        client = TonicOllamaClient()
        
        with pytest.raises(ValueError, match="does not exist"):
            client.get_conversation("nonexistent")
        
        with pytest.raises(ValueError, match="does not exist"):
            client.clear_conversation("nonexistent")
        
        with pytest.raises(ValueError, match="does not exist"):
            client.delete_conversation("nonexistent")
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_exhaustion(self, mock_client_with_errors):
        """Test that retry mechanism eventually gives up."""
        client, mock_ollama, mock_server_check = mock_client_with_errors
        
        # Ensure ensure_server_ready thinks server is up initially
        mock_server_check.return_value = True
        
        mock_ollama.chat.side_effect = ConnectionError("Persistent connection error")
        
        with pytest.raises(ConnectionError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="test")
        
        # API_RETRY_CONFIG has stop_after_attempt(3)
        assert mock_ollama.chat.call_count == 3 
    
    @pytest.mark.asyncio
    async def test_malformed_api_responses(self, mock_client_with_errors):
        """Test handling of malformed API responses."""
        client, mock_ollama, _ = mock_client_with_errors
        
        mock_ollama.chat.return_value = {"invalid": "response"}
        
        with pytest.raises(KeyError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="test")
        
        mock_ollama.chat.reset_mock()
        mock_ollama.chat.return_value = {"message": {"role": "assistant", "content": "Valid chat response"}}

        mock_ollama.embeddings.return_value = {"invalid": "response"}
        
        with pytest.raises(KeyError):
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="test")

        mock_ollama.embeddings.reset_mock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1,0.2]}
        
    @pytest.mark.asyncio
    async def test_unexpected_exceptions(self, mock_client_with_errors):
        """Test handling of unexpected exceptions."""
        client, mock_ollama, _ = mock_client_with_errors
        
        mock_ollama.chat.side_effect = RuntimeError("Unexpected error")
        
        with pytest.raises(RuntimeError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="test")
        
        mock_ollama.embeddings.side_effect = ValueError("Unexpected value error")
        
        with pytest.raises(ValueError):
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="test")
    
    @pytest.mark.asyncio
    async def test_ensure_server_ready_server_not_running(self, mock_client_with_errors):
        """Test ensure_server_ready when server is initially not running."""
        client, _, mock_server_check = mock_client_with_errors
        
        mock_server_check.return_value = False # Simulate server is not running
        client.max_server_startup_attempts = 2 # Set for the test
        
        with pytest.raises(OllamaServerNotRunningError):
            await client.ensure_server_ready()
        
        assert mock_server_check.call_count == 2

    @pytest.mark.asyncio
    async def test_ensure_server_ready_becomes_available(self, mock_client_with_errors):
        """Test ensure_server_ready when server becomes available after an attempt."""
        client, _, mock_server_check = mock_client_with_errors

        # Simulate server not running on first call, then running on second
        mock_server_check.side_effect = [False, True]
        client.max_server_startup_attempts = 2

        await client.ensure_server_ready() # Should pass
        assert mock_server_check.call_count == 2
