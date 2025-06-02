import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from tonic_ollama_client import TonicOllamaClient, ResponseError, ModelNotReadyError
import asyncio

APPROVED_MODELS = ["llama3.1:latest", "phi4:latest", "qwen2:7b"]
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
            mock_instance.list = AsyncMock()
            mock_instance.pull = AsyncMock()
            
            client = TonicOllamaClient(debug=True)
            yield client, mock_instance
    
    @pytest.mark.asyncio
    async def test_chat_connection_error(self, mock_client_with_errors):
        """Test chat with connection error."""
        client, mock_ollama = mock_client_with_errors
        
        mock_ollama.chat.side_effect = ConnectionError("Connection refused")
        
        with pytest.raises(ConnectionError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="test")
    
    @pytest.mark.asyncio
    async def test_chat_timeout_error(self, mock_client_with_errors):
        """Test chat with timeout error."""
        client, mock_ollama = mock_client_with_errors
        
        mock_ollama.chat.side_effect = TimeoutError("Request timed out")
        
        with pytest.raises(TimeoutError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="test")
    
    @pytest.mark.asyncio
    async def test_chat_response_error_various_codes(self, mock_client_with_errors):
        """Test chat with various HTTP response errors."""
        client, mock_ollama = mock_client_with_errors
        
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
        client, mock_ollama = mock_client_with_errors
        
        mock_ollama.embeddings.side_effect = ConnectionError("Network unreachable")
        
        with pytest.raises(ConnectionError):
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="test")
    
    @pytest.mark.asyncio
    async def test_embedding_response_error(self, mock_client_with_errors):
        """Test embedding generation with response error."""
        client, mock_ollama = mock_client_with_errors
        
        mock_ollama.embeddings.side_effect = ResponseError("Embedding failed", 422)
        
        with pytest.raises(ResponseError) as exc_info:
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="test")
        
        assert exc_info.value.status_code == 422
    
    @pytest.mark.asyncio
    async def test_check_model_ready_list_error(self, mock_client_with_errors):
        """Test check_model_ready with list API error."""
        client, mock_ollama = mock_client_with_errors
        client.max_readiness_attempts = 1

        mock_ollama.list.side_effect = ResponseError("List failed", 503)
        mock_ollama.chat.return_value = {"message": {"content": "Simulated chat response"}}
        
        with pytest.raises(ModelNotReadyError):
            await client.check_model_ready("phi4:latest")
    
    @pytest.mark.asyncio
    async def test_check_model_ready_pull_error(self, mock_client_with_errors):
        """Test check_model_ready with pull API error."""
        client, mock_ollama = mock_client_with_errors
        client.max_readiness_attempts = 1

        mock_ollama.list.return_value = {"models": []}
        mock_ollama.pull.side_effect = ResponseError("Pull failed", 404)
        mock_ollama.chat.return_value = {"message": {"content": "Simulated chat response"}}
        
        with pytest.raises(ModelNotReadyError):
            await client.check_model_ready("qwen2:7b")
    
    @pytest.mark.asyncio
    async def test_check_model_ready_chat_error(self, mock_client_with_errors):
        """Test check_model_ready with chat API error during responsiveness check."""
        client, mock_ollama = mock_client_with_errors
        
        mock_ollama.list.return_value = {
            "models": [{"name": "llama3.1:latest"}]
        }
        
        mock_ollama.chat.side_effect = ResponseError("Chat failed", 500)
        
        with pytest.raises(ModelNotReadyError):
            await client.check_model_ready("llama3.1:latest")
    
    @pytest.mark.asyncio
    async def test_model_not_ready_error_details(self, mock_client_with_errors):
        """Test ModelNotReadyError with specific details."""
        client, mock_ollama = mock_client_with_errors
        client.max_readiness_attempts = 1

        mock_ollama.list.return_value = {
            "models": [{"name": "llama3.1:latest"}]
        }
        mock_ollama.chat.return_value = {
            "message": {"content": "I'm not ready"}
        }

        with pytest.raises(ModelNotReadyError) as exc_info:
            await client.check_model_ready("llama3.1:latest")
        
        error = exc_info.value
        assert error.model_name == "llama3.1:latest"
        assert error.status_code == 400
        assert "llama3.1:latest" in str(error)
        assert "Failed to make model ready after multiple attempts" in str(error)
        mock_ollama.chat.assert_called_once()

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
        client, mock_ollama = mock_client_with_errors
        
        mock_ollama.chat.side_effect = ConnectionError("Persistent connection error")
        
        with pytest.raises(ConnectionError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="test")
        
        assert mock_ollama.chat.call_count > 1
    
    @pytest.mark.asyncio
    async def test_malformed_api_responses(self, mock_client_with_errors):
        """Test handling of malformed API responses."""
        client, mock_ollama = mock_client_with_errors
        
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
        
        mock_ollama.list.side_effect = TypeError("List call returned unexpected data type")
        
        mock_ollama.pull.return_value = {"status": "success"} 

        client.max_readiness_attempts = 1
        
        with pytest.raises(ModelNotReadyError):
             await client.check_model_ready(DEFAULT_TEST_MODEL)
    
    @pytest.mark.asyncio
    async def test_unexpected_exceptions(self, mock_client_with_errors):
        """Test handling of unexpected exceptions."""
        client, mock_ollama = mock_client_with_errors
        
        mock_ollama.chat.side_effect = RuntimeError("Unexpected error")
        
        with pytest.raises(RuntimeError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="test")
        
        mock_ollama.embeddings.side_effect = ValueError("Unexpected value error")
        
        with pytest.raises(ValueError):
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="test")
    
    @pytest.mark.asyncio
    async def test_partial_failures_in_check_model_ready(self, mock_client_with_errors):
        """Test partial failures in check_model_ready workflow."""
        client, mock_ollama = mock_client_with_errors
        client.max_readiness_attempts = 2

        list_call_count = 0
        def list_side_effect(*args, **kwargs):
            nonlocal list_call_count
            list_call_count += 1
            if list_call_count == 1:
                return {"models": []}
            else:
                return {"models": [{"name": "llama3.1:latest"}]}

        chat_call_count = 0
        def chat_side_effect(*args, **kwargs):
            nonlocal chat_call_count
            chat_call_count += 1
            if chat_call_count == 1:
                return {"message": {"content": "STILL LOADING..."}}
            else:
                return {"message": {"content": "READY"}}


        mock_ollama.list.side_effect = list_side_effect
        mock_ollama.pull.side_effect = ResponseError("Pull failed", 404)
        mock_ollama.chat.side_effect = chat_side_effect

        await client.check_model_ready("llama3.1:latest")

        assert mock_ollama.list.call_count >= 2
        mock_ollama.pull.assert_called()
        assert mock_ollama.chat.call_count >= 2
