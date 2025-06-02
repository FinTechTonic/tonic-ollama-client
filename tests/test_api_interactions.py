import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from tonic_ollama_client import TonicOllamaClient, ResponseError, ModelNotReadyError

APPROVED_MODELS = ["llama3.1:latest", "phi4:latest", "qwen2:7b"]
DEFAULT_TEST_MODEL = "llama3.1:latest"

@pytest_asyncio.fixture
async def mock_client():
    """Provides a mocked TonicOllamaClient and its underlying Ollama AsyncClient mock."""
    with patch('tonic_ollama_client.AsyncClient') as MockedOllamaAsyncClient:
        mock_ollama_instance = MockedOllamaAsyncClient.return_value
        
        mock_ollama_instance.chat = AsyncMock()
        mock_ollama_instance.embeddings = AsyncMock()
        mock_ollama_instance.list = AsyncMock()
        mock_ollama_instance.pull = AsyncMock()
        
        toc_client = TonicOllamaClient(debug=True)
        yield toc_client, mock_ollama_instance

@pytest.mark.asyncio
async def test_chat_basic(mock_client):
    """Test basic chat functionality."""
    client, mock_ollama = mock_client
    
    mock_response = {
        "message": {
            "role": "assistant",
            "content": "I am an AI assistant."
        }
    }
    mock_ollama.chat.return_value = mock_response
    
    response = await client.chat(
        model=DEFAULT_TEST_MODEL,
        message="Hello, who are you?",
        system_prompt="You are a helpful assistant"
    )
    
    assert response == mock_response
    
    # Verify conversation management
    conversations = client.list_conversations()
    assert len(conversations) == 1
    
    conv_id = conversations[0]
    messages = client.get_conversation(conv_id)
    assert len(messages) == 2 
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, who are you?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "I am an AI assistant."

@pytest.mark.asyncio
async def test_chat_with_existing_conversation(mock_client):
    """Test chat with pre-existing conversation ID."""
    client, mock_ollama = mock_client
    
    # Create a conversation first
    conv_id = await client.create_conversation("test-conv-123")
    
    mock_response = {
        "message": {
            "role": "assistant",
            "content": "Hello there!"
        }
    }
    mock_ollama.chat.return_value = mock_response
    
    response = await client.chat(
        model=DEFAULT_TEST_MODEL,
        message="Hi",
        conversation_id=conv_id
    )
    
    assert response == mock_response
    
    # Verify the specific conversation was used
    messages = client.get_conversation(conv_id)
    assert len(messages) == 2
    assert messages[0]["content"] == "Hi"
    assert messages[1]["content"] == "Hello there!"

@pytest.mark.asyncio
async def test_chat_without_system_prompt(mock_client):
    """Test chat without system prompt."""
    client, mock_ollama = mock_client
    
    mock_response = {
        "message": {
            "role": "assistant",
            "content": "Response without system prompt"
        }
    }
    mock_ollama.chat.return_value = mock_response
    
    response = await client.chat(
        model=DEFAULT_TEST_MODEL,
        message="Test message"
    )
    
    assert response == mock_response
    
    # Verify API was called correctly without system prompt
    mock_ollama.chat.assert_called_once()
    call_args = mock_ollama.chat.call_args[1]
    messages = call_args['messages']
    assert len(messages) == 1
    assert messages[0]['role'] == 'user'
    assert messages[0]['content'] == 'Test message'

@pytest.mark.asyncio
async def test_chat_with_temperature(mock_client):
    """Test chat with custom temperature."""
    client, mock_ollama = mock_client
    
    mock_response = {
        "message": {
            "role": "assistant",
            "content": "Response with custom temperature"
        }
    }
    mock_ollama.chat.return_value = mock_response
    
    response = await client.chat(
        model=DEFAULT_TEST_MODEL,
        message="Test",
        temperature=0.9
    )
    
    assert response == mock_response
    
    # Verify temperature was passed correctly
    mock_ollama.chat.assert_called_once()
    call_args = mock_ollama.chat.call_args[1]
    assert call_args['options']['temperature'] == 0.9

@pytest.mark.asyncio
async def test_chat_error_handling(mock_client):
    """Test chat error handling."""
    client, mock_ollama = mock_client
    
    # Test ResponseError
    mock_ollama.chat.side_effect = ResponseError("Model not found", 404)
    
    with pytest.raises(ResponseError) as exc_info:
        await client.chat(model="nonexistent", message="test")
    
    assert "Model not found" in str(exc_info.value)
    assert exc_info.value.status_code == 404

@pytest.mark.asyncio
async def test_generate_embedding_basic(mock_client):
    """Test basic embedding generation."""
    client, mock_ollama = mock_client
    
    mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_ollama.embeddings.return_value = {"embedding": mock_embedding}
    
    embedding = await client.generate_embedding(
        model=DEFAULT_TEST_MODEL,
        text="This is a test"
    )
    
    assert embedding == mock_embedding
    
    # Verify API was called correctly
    mock_ollama.embeddings.assert_called_once_with(
        model=DEFAULT_TEST_MODEL,
        prompt="This is a test"
    )

@pytest.mark.asyncio
async def test_generate_embedding_error_handling(mock_client):
    """Test embedding generation error handling."""
    client, mock_ollama = mock_client
    
    # Test ResponseError
    mock_ollama.embeddings.side_effect = ResponseError("Embedding failed", 500)
    
    with pytest.raises(ResponseError) as exc_info:
        await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="test")
    
    assert "Embedding failed" in str(exc_info.value)
    assert exc_info.value.status_code == 500

@pytest.mark.asyncio
async def test_check_model_ready_found_locally(mock_client):
    """Test check_model_ready when model is found locally."""
    client, mock_ollama = mock_client
    
    # Mock list response showing model exists
    mock_ollama.list.return_value = {
        "models": [
            {"name": "llama3.1:latest", "size": 1000000}
        ]
    }
    
    # Mock successful chat response
    mock_ollama.chat.return_value = {
        "message": {"content": "READY"}
    }
    
    # Should not raise an exception
    await client.check_model_ready("llama3.1:latest")
    
    # Verify list was called but pull was not
    mock_ollama.list.assert_called_once()
    mock_ollama.pull.assert_not_called()
    mock_ollama.chat.assert_called_once()

@pytest.mark.asyncio
async def test_check_model_ready_needs_pull(mock_client):
    """Test check_model_ready when model needs to be pulled."""
    client, mock_ollama = mock_client
    
    # Mock list response showing model doesn't exist
    mock_ollama.list.return_value = {"models": []}
    
    # Mock successful pull
    mock_ollama.pull.return_value = {"status": "success"}
    
    # Mock successful chat response
    mock_ollama.chat.return_value = {
        "message": {"content": "READY"}
    }
    
    # Should not raise an exception
    await client.check_model_ready("phi4:latest")
    
    # Verify pull was called
    mock_ollama.list.assert_called_once()
    mock_ollama.pull.assert_called_once_with(model="phi4:latest", stream=False)
    mock_ollama.chat.assert_called_once()

@pytest.mark.asyncio
async def test_check_model_ready_model_not_responsive(mock_client):
    """Test check_model_ready when model doesn't respond with READY."""
    client, mock_ollama = mock_client
    
    # Mock list response showing model exists
    mock_ollama.list.return_value = {
        "models": [{"name": "qwen2:7b"}]
    }
    
    # Mock chat responses that don't contain "READY"
    mock_ollama.chat.return_value = {
        "message": {"content": "Hello there"}
    }
    
    # Should raise ModelNotReadyError after max attempts
    with pytest.raises(ModelNotReadyError) as exc_info:
        await client.check_model_ready("qwen2:7b")
    
    assert "qwen2:7b" in str(exc_info.value)

@pytest.mark.asyncio
async def test_check_model_ready_pull_fails(mock_client):
    """Test check_model_ready when pull fails."""
    client, mock_ollama = mock_client
    
    # Mock list response showing model doesn't exist
    mock_ollama.list.return_value = {"models": []}
    
    # Mock failed pull
    mock_ollama.pull.side_effect = ResponseError("Pull failed", 404)
    
    # Should raise ModelNotReadyError after attempts
    with pytest.raises(ModelNotReadyError):
        await client.check_model_ready("phi4:latest")

@pytest.mark.asyncio
async def test_multiple_conversations(mock_client):
    """Test managing multiple conversations."""
    client, mock_ollama = mock_client
    
    mock_response = {
        "message": {
            "role": "assistant",
            "content": "Response"
        }
    }
    mock_ollama.chat.return_value = mock_response
    
    # Create multiple conversations
    conv1 = await client.create_conversation("conv1")
    conv2 = await client.create_conversation("conv2")
    
    # Chat in each conversation
    await client.chat(model=DEFAULT_TEST_MODEL, message="Message 1", conversation_id=conv1)
    await client.chat(model=DEFAULT_TEST_MODEL, message="Message 2", conversation_id=conv2)
    
    # Verify conversations are separate
    assert len(client.list_conversations()) == 2
    assert client.get_conversation(conv1)[0]["content"] == "Message 1"
    assert client.get_conversation(conv2)[0]["content"] == "Message 2"

@pytest.mark.asyncio
async def test_client_debug_mode(mock_client):
    """Test client debug output."""
    client, mock_ollama = mock_client
    
    # Verify debug mode is enabled
    assert client.debug is True
    
    mock_response = {
        "message": {
            "role": "assistant",
            "content": "Debug test"
        }
    }
    mock_ollama.chat.return_value = mock_response
    
    # This should trigger debug prints
    await client.chat(model=DEFAULT_TEST_MODEL, message="test")
    
    # Verify the call was made (debug prints are to console)
    mock_ollama.chat.assert_called_once()

@pytest.mark.asyncio
async def test_async_client_reuse():
    """Test that async client is reused across calls."""
    client = TonicOllamaClient()
    
    # Get client multiple times
    async_client1 = client.get_async_client()
    async_client2 = client.get_async_client()
    
    # Should be the same instance
    assert async_client1 is async_client2
