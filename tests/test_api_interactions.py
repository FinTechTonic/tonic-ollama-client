import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from tonic_ollama_client import TonicOllamaClient, ResponseError, OllamaServerNotRunningError
from ollama._types import ChatResponse, Message # Import necessary types
from typing import AsyncGenerator, Dict, Any # For type hinting

APPROVED_MODELS = ["llama3.1:latest", "phi4:latest", "qwen3:8b", "mistral:latest"]
DEFAULT_TEST_MODEL = "llama3.1:latest"

@pytest_asyncio.fixture
async def mock_client():
    """Provides a mocked TonicOllamaClient and its underlying Ollama AsyncClient mock."""
    with patch('tonic_ollama_client.AsyncClient') as MockedOllamaAsyncClient:
        mock_ollama_instance = MockedOllamaAsyncClient.return_value
        
        mock_ollama_instance.chat = AsyncMock()
        mock_ollama_instance.embeddings = AsyncMock()
        
        # Patch _is_ollama_server_running_sync for the client instance for all tests using this fixture
        with patch.object(TonicOllamaClient, '_is_ollama_server_running_sync', return_value=True):
            toc_client = TonicOllamaClient(debug=True)
            yield toc_client, mock_ollama_instance

@pytest.mark.asyncio
async def test_chat_basic(mock_client):
    """Test basic chat functionality (non-streaming)."""
    client, mock_ollama = mock_client
    
    mock_content = "I am an AI assistant."
    # The underlying ollama client's chat method returns a ChatResponse object
    mock_ollama.chat.return_value = ChatResponse(
        model=DEFAULT_TEST_MODEL,
        created_at="dummy_timestamp",
        message=Message(role="assistant", content=mock_content),
        done=True
    )
    
    # TonicOllamaClient.chat (non-streaming) returns a dictionary (from model_dump)
    response_dict = await client.chat(
        model=DEFAULT_TEST_MODEL,
        message="Hello, who are you?",
        system_prompt="You are a helpful assistant",
        stream=False # Explicitly non-streaming
    )
    
    assert isinstance(response_dict, dict)
    assert response_dict["message"]["content"] == mock_content
    assert response_dict["message"]["role"] == "assistant"
    assert response_dict["model"] == DEFAULT_TEST_MODEL
    
    # Verify conversation management
    conversations = client.list_conversations()
    assert len(conversations) == 1
    
    conv_id = conversations[0]
    messages = client.get_conversation(conv_id)
    assert len(messages) == 2 
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, who are you?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == mock_content

@pytest.mark.asyncio
async def test_chat_streaming(mock_client):
    """Test chat functionality with streaming."""
    client, mock_ollama = mock_client

    # Mock the underlying ollama client's chat to return an async generator
    async def mock_ollama_stream_generator():
        yield ChatResponse(model=DEFAULT_TEST_MODEL, created_at="ts1", message=Message(role="assistant", content="Hello "), done=False)
        yield ChatResponse(model=DEFAULT_TEST_MODEL, created_at="ts2", message=Message(role="assistant", content="World!"), done=False)
        # The official ollama client might send a final message with empty content but done=True
        yield ChatResponse(model=DEFAULT_TEST_MODEL, created_at="ts3", message=Message(role="assistant", content=""), done=True)

    mock_ollama.chat.return_value = mock_ollama_stream_generator()

    full_response_content = ""
    stream_chunk_count = 0

    # Call TonicOllamaClient's chat method with stream=True
    response_generator = await client.chat(
        model=DEFAULT_TEST_MODEL,
        message="Stream test message",
        stream=True
    )

    assert isinstance(response_generator, AsyncGenerator)

    async for chunk in response_generator:
        stream_chunk_count += 1
        assert isinstance(chunk, ChatResponse)
        if chunk.message and chunk.message.content:
            full_response_content += chunk.message.content
        # No break here, consume the whole stream to test history update
    
    assert full_response_content == "Hello World!"
    assert stream_chunk_count == 3 # Based on the mock_ollama_stream_generator

    # Verify conversation history was updated correctly after stream
    # Assumes client.chat with stream=True creates a conversation if ID not provided
    conv_id = client.list_conversations()[0] 
    messages = client.get_conversation(conv_id)
    
    assert len(messages) == 2 # User message + full assistant message
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Stream test message"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hello World!" # Accumulated content

    # Check that the underlying mock_ollama.chat was called once with stream=True
    mock_ollama.chat.assert_called_once()
    args, kwargs = mock_ollama.chat.call_args
    assert kwargs.get('stream') is True


@pytest.mark.asyncio
async def test_chat_with_existing_conversation(mock_client):
    """Test chat with pre-existing conversation ID (non-streaming)."""
    client, mock_ollama = mock_client
    
    conv_id = await client.create_conversation("test-conv-123")
    
    mock_content = "Hello there!"
    mock_ollama.chat.return_value = ChatResponse(
        model=DEFAULT_TEST_MODEL,
        created_at="dummy_timestamp",
        message=Message(role="assistant", content=mock_content),
        done=True
    )
    
    response_dict = await client.chat(
        model=DEFAULT_TEST_MODEL,
        message="Hi",
        conversation_id=conv_id,
        stream=False # Explicitly non-streaming
    )
    
    assert response_dict["message"]["content"] == mock_content
    
    messages = client.get_conversation(conv_id)
    assert len(messages) == 2
    assert messages[0]["content"] == "Hi"
    assert messages[1]["content"] == mock_content

@pytest.mark.asyncio
async def test_chat_without_system_prompt(mock_client):
    """Test chat without system prompt (non-streaming)."""
    client, mock_ollama = mock_client
    
    mock_content = "Response without system prompt"
    mock_ollama.chat.return_value = ChatResponse(
        model=DEFAULT_TEST_MODEL,
        created_at="dummy_timestamp",
        message=Message(role="assistant", content=mock_content),
        done=True
    )
    
    response_dict = await client.chat(
        model=DEFAULT_TEST_MODEL,
        message="No system prompt here",
        stream=False # Explicitly non-streaming
    )
    
    assert response_dict["message"]["content"] == mock_content
    messages = client.get_conversation(client.list_conversations()[0])
    assert len(messages) == 2 
    assert messages[0]["role"] == "user"

@pytest.mark.asyncio
async def test_chat_with_temperature(mock_client):
    """Test chat with custom temperature (non-streaming)."""
    client, mock_ollama = mock_client
    
    mock_content = "Response with custom temperature"
    mock_ollama.chat.return_value = ChatResponse(
        model=DEFAULT_TEST_MODEL,
        created_at="dummy_timestamp",
        message=Message(role="assistant", content=mock_content),
        done=True
    )
    
    await client.chat(
        model=DEFAULT_TEST_MODEL,
        message="Test temperature",
        temperature=0.5,
        stream=False # Explicitly non-streaming
    )
    
    mock_ollama.chat.assert_called_once()
    args, kwargs = mock_ollama.chat.call_args
    assert kwargs['options']['temperature'] == 0.5

@pytest.mark.asyncio
async def test_chat_error_handling(mock_client):
    """Test error handling in chat method."""
    client, mock_ollama = mock_client
    
    mock_ollama.chat.side_effect = ResponseError("API Error", 500)
    
    with pytest.raises(ResponseError):
        await client.chat(model=DEFAULT_TEST_MODEL, message="Error test", stream=False)

@pytest.mark.asyncio
async def test_generate_embedding_basic(mock_client):
    """Test basic embedding generation."""
    client, mock_ollama = mock_client
    
    mock_embedding_data = [0.1, 0.2, 0.3, 0.4, 0.5]
    # The ollama client's embeddings method returns a dict, not a Pydantic model directly for this one
    mock_ollama.embeddings.return_value = {"embedding": mock_embedding_data}
    
    embedding = await client.generate_embedding(
        model=DEFAULT_TEST_MODEL,
        text="Embed this text"
    )
    
    assert embedding == mock_embedding_data
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
    client, _ = mock_client 
    
    # The mock_client fixture already patches _is_ollama_server_running_sync to return True
    await client.ensure_server_ready() 
    # To assert the call on the mock provided by the fixture, we need to access it.
    # This is a bit tricky as the patch is on the class.
    # For simplicity, we'll trust the fixture sets it up correctly.
    # If specific call count for _is_ollama_server_running_sync is needed,
    # the patch in the fixture should be yielded or re-patched here.

@pytest.mark.asyncio
async def test_ensure_server_ready_server_not_responsive(mock_client):
    """Test ensure_server_ready when server doesn't respond."""
    client, _ = mock_client 
    client.max_server_startup_attempts = 2

    # We need to override the fixture's patch for this specific test
    with patch.object(TonicOllamaClient, '_is_ollama_server_running_sync', return_value=False) as mock_is_running:
        with pytest.raises(OllamaServerNotRunningError):
            await client.ensure_server_ready()
        assert mock_is_running.call_count == client.max_server_startup_attempts

@pytest.mark.asyncio
async def test_ensure_server_ready_becomes_responsive(mock_client):
    """Test ensure_server_ready when server becomes responsive after initial failure."""
    client, _ = mock_client
    client.max_server_startup_attempts = 3

    # Override the fixture's patch for this specific test
    with patch.object(TonicOllamaClient, '_is_ollama_server_running_sync', side_effect=[False, True, True]) as mock_is_running:
        await client.ensure_server_ready()
        assert mock_is_running.call_count == 2


@pytest.mark.asyncio
async def test_multiple_conversations(mock_client):
    """Test managing multiple conversations (non-streaming)."""
    client, mock_ollama = mock_client
    
    mock_content = "Response"
    mock_ollama.chat.return_value = ChatResponse(
        model=DEFAULT_TEST_MODEL,
        created_at="dummy_timestamp",
        message=Message(role="assistant", content=mock_content),
        done=True
    )
    
    conv1 = await client.create_conversation("conv1")
    conv2 = await client.create_conversation("conv2")
    
    await client.chat(model=DEFAULT_TEST_MODEL, message="Message 1", conversation_id=conv1, stream=False)
    # Reset mock for the next call if return_value needs to be different or call_args checked per call
    # For this test, same return value is fine.
    await client.chat(model=DEFAULT_TEST_MODEL, message="Message 2", conversation_id=conv2, stream=False)
    
    assert len(client.list_conversations()) == 2
    # Content of user message
    assert client.get_conversation(conv1)[0]["content"] == "Message 1"
    # Content of assistant message (second message in history)
    assert client.get_conversation(conv1)[1]["content"] == mock_content 
    
    assert client.get_conversation(conv2)[0]["content"] == "Message 2"
    assert client.get_conversation(conv2)[1]["content"] == mock_content


@pytest.mark.asyncio
async def test_client_debug_mode(mock_client):
    """Test client debug output (non-streaming)."""
    client, mock_ollama = mock_client
    
    assert client.debug is True # Set by mock_client fixture
    
    mock_content = "Debug test"
    mock_ollama.chat.return_value = ChatResponse(
        model=DEFAULT_TEST_MODEL,
        created_at="dummy_timestamp",
        message=Message(role="assistant", content=mock_content),
        done=True
    )
    
    await client.chat(model=DEFAULT_TEST_MODEL, message="test", stream=False)
    
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
