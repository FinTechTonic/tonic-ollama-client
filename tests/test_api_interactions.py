import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from tonic_ollama_client import TonicOllamaClient, ResponseError, OllamaServerNotRunningError

APPROVED_MODELS = ["llama3.1:latest", "phi4:latest", "qwen2:7b", "mistral:latest"]
DEFAULT_TEST_MODEL = "llama3.1:latest"

@pytest_asyncio.fixture
async def mock_client():
    """Provides a mocked TonicOllamaClient and its underlying Ollama AsyncClient mock."""
    with patch('tonic_ollama_client.AsyncClient') as MockedOllamaAsyncClient:
        mock_ollama_instance = MockedOllamaAsyncClient.return_value
        
        mock_ollama_instance.chat = AsyncMock()
        mock_ollama_instance.embeddings = AsyncMock()
        
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
    
    # Patch _is_ollama_server_running_sync to simulate server being ready
    with patch.object(client, '_is_ollama_server_running_sync', return_value=True):
        response = await client.chat(
            model=DEFAULT_TEST_MODEL,
            message="No system prompt here"
        )
    
    assert response == mock_response
    messages = client.get_conversation(client.list_conversations()[0])
    assert len(messages) == 2 # User and assistant
    assert messages[0]["role"] == "user"

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
    
    # Patch _is_ollama_server_running_sync to simulate server being ready
    with patch.object(client, '_is_ollama_server_running_sync', return_value=True):
        await client.chat(
            model=DEFAULT_TEST_MODEL,
            message="Test temperature",
            temperature=0.5
        )
    
    mock_ollama.chat.assert_called_once()
    args, kwargs = mock_ollama.chat.call_args
    assert kwargs['options']['temperature'] == 0.5

@pytest.mark.asyncio
async def test_chat_error_handling(mock_client):
    """Test error handling in chat method."""
    client, mock_ollama = mock_client
    
    # Test ResponseError
    mock_ollama.chat.side_effect = ResponseError("API Error", 500)
    
    with pytest.raises(ResponseError):
        # Patch _is_ollama_server_running_sync to simulate server being ready
        with patch.object(client, '_is_ollama_server_running_sync', return_value=True):
            await client.chat(model=DEFAULT_TEST_MODEL, message="Error test")

@pytest.mark.asyncio
async def test_generate_embedding_basic(mock_client):
    """Test basic embedding generation."""
    client, mock_ollama = mock_client
    
    mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_ollama.embeddings.return_value = {"embedding": mock_embedding}
    
    # Patch _is_ollama_server_running_sync to simulate server being ready
    with patch.object(client, '_is_ollama_server_running_sync', return_value=True):
        embedding = await client.generate_embedding(
            model=DEFAULT_TEST_MODEL,
            text="Embed this text"
        )
    
    assert embedding == mock_embedding # Use the mock_embedding variable for assertion
    mock_ollama.embeddings.assert_called_once_with(model=DEFAULT_TEST_MODEL, prompt="Embed this text")

@pytest.mark.asyncio
async def test_generate_embedding_error_handling(mock_client):
    """Test error handling in embedding generation."""
    client, mock_ollama = mock_client
    
    # Test ResponseError
    mock_ollama.embeddings.side_effect = ResponseError("Embedding Error", 500)
    
    with pytest.raises(ResponseError):
        # Patch _is_ollama_server_running_sync to simulate server being ready
        with patch.object(client, '_is_ollama_server_running_sync', return_value=True):
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="Error embed")

# The following tests are for ensure_server_ready, replacing check_model_ready tests
@pytest.mark.asyncio
async def test_ensure_server_ready_server_is_responsive(mock_client):
    """Test ensure_server_ready when server responds correctly."""
    client, _ = mock_client # We don't need mock_ollama for this specific test
    
    # Patch _is_ollama_server_running_sync to simulate server being ready
    with patch.object(client, '_is_ollama_server_running_sync', return_value=True) as mock_is_running:
        await client.ensure_server_ready() # Changed from check_model_ready
        mock_is_running.assert_called_once()

@pytest.mark.asyncio
async def test_ensure_server_ready_server_not_responsive(mock_client):
    """Test ensure_server_ready when server doesn't respond."""
    client, _ = mock_client # We don't need mock_ollama for this specific test
    client.max_server_startup_attempts = 2 # For faster test

    # Patch _is_ollama_server_running_sync to simulate server NOT being ready
    with patch.object(client, '_is_ollama_server_running_sync', return_value=False) as mock_is_running:
        with pytest.raises(OllamaServerNotRunningError):
            await client.ensure_server_ready() # Changed from check_model_ready
        assert mock_is_running.call_count == client.max_server_startup_attempts

@pytest.mark.asyncio
async def test_ensure_server_ready_becomes_responsive(mock_client):
    """Test ensure_server_ready when server becomes responsive after initial failure."""
    client, _ = mock_client
    client.max_server_startup_attempts = 3

    # Simulate server becoming responsive on the second attempt
    with patch.object(client, '_is_ollama_server_running_sync', side_effect=[False, True, True]) as mock_is_running:
        await client.ensure_server_ready()
        assert mock_is_running.call_count == 2


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
