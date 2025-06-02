import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from tonic_ollama_client import TonicOllamaClient, CONCURRENT_MODELS
from ollama._types import ChatResponse, Message # Import necessary types

@pytest.mark.asyncio
async def test_sequential_model_access():
    """Test that model access is properly serialized with the semaphore."""
    # Ensure we're testing with CONCURRENT_MODELS=1
    assert CONCURRENT_MODELS == 1, "This test assumes CONCURRENT_MODELS=1"
    
    client = TonicOllamaClient(debug=True)
    
    # Create a mock for the AsyncClient's chat method
    with patch('tonic_ollama_client.AsyncClient') as MockAsyncClient:
        mock_instance = MockAsyncClient.return_value
        
        # Configure the mock chat method to have a delay
        async def delayed_chat(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate model loading/processing time
            # Return a ChatResponse object instead of a raw dict
            return ChatResponse(
                model="model1", 
                created_at="dummy_ts", 
                message=Message(role="assistant", content="Response"), 
                done=True
            )
        
        mock_instance.chat = AsyncMock(side_effect=delayed_chat)
        
        # Patch _is_ollama_server_running_sync to avoid external calls
        with patch.object(client, '_is_ollama_server_running_sync', return_value=True):
            # Start multiple sequential chat requests
            start_time = asyncio.get_event_loop().time()
            
            async def wrapped_chat_call(model_name: str, msg_content: str):
                # This is the call Pylance was flagging.
                # The TonicOllamaClient.chat method does have a 'message' parameter.
                return await client.chat(model=model_name, message=msg_content)

            task1 = asyncio.create_task(wrapped_chat_call(model_name="model1", msg_content="message1"))
            task2 = asyncio.create_task(wrapped_chat_call(model_name="model1", msg_content="message2"))
            
            # Wait for both tasks to complete
            await asyncio.gather(task1, task2)
            
            end_time = asyncio.get_event_loop().time()
            
            # With CONCURRENT_MODELS=1, the second request should wait for the first to complete
            # So the total time should be at least 1 second (2 * 0.5s)
            assert end_time - start_time >= 0.9, "Requests should be processed sequentially with the semaphore"
            
            # Check that the chat method was called twice
            assert mock_instance.chat.call_count == 2

@pytest.mark.asyncio
async def test_chat_and_embedding_sequential():
    """Test that chat and generate_embedding requests are properly serialized."""
    # Ensure we're testing with CONCURRENT_MODELS=1
    assert CONCURRENT_MODELS == 1, "This test assumes CONCURRENT_MODELS=1"
    
    client = TonicOllamaClient(debug=True)
    
    # Create mocks for the AsyncClient's methods
    with patch('tonic_ollama_client.AsyncClient') as MockAsyncClient:
        mock_instance = MockAsyncClient.return_value
        
        # Configure the mock methods to have a delay
        async def delayed_chat(*args, **kwargs):
            await asyncio.sleep(0.5)
            # Return a ChatResponse object
            return ChatResponse(
                model="model1",
                created_at="dummy_ts",
                message=Message(role="assistant", content="Response"),
                done=True
            )
        
        async def delayed_embeddings(*args, **kwargs):
            await asyncio.sleep(0.5)
            return {"embedding": [0.1, 0.2, 0.3]}
        
        mock_instance.chat = AsyncMock(side_effect=delayed_chat)
        mock_instance.embeddings = AsyncMock(side_effect=delayed_embeddings)
        
        # Patch _is_ollama_server_running_sync to avoid external calls
        with patch.object(client, '_is_ollama_server_running_sync', return_value=True):
            # Start sequential chat and embedding requests
            start_time = asyncio.get_event_loop().time()

            async def wrapped_chat_call(model_name: str, msg_content: str):
                # This is the call Pylance was flagging.
                return await client.chat(model=model_name, message=msg_content)

            async def wrapped_embedding_call(model_name: str, text_content: str):
                return await client.generate_embedding(model=model_name, text=text_content)

            task1 = asyncio.create_task(wrapped_chat_call(model_name="model1", msg_content="message1"))
            task2 = asyncio.create_task(wrapped_embedding_call(model_name="model1", text_content="text2"))

            # Wait for both tasks to complete
            await asyncio.gather(task1, task2)
            
            end_time = asyncio.get_event_loop().time()
            
            # With CONCURRENT_MODELS=1, the second request should wait for the first to complete
            assert end_time - start_time >= 0.9, "Different method requests should be processed sequentially"
            
            # Check that both methods were called
            assert mock_instance.chat.call_count == 1
            assert mock_instance.embeddings.call_count == 1

@pytest.mark.asyncio
async def test_semaphore_value():
    """Test that the semaphore is initialized with CONCURRENT_MODELS=1."""
    # Ensure we're testing with CONCURRENT_MODELS=1
    assert CONCURRENT_MODELS == 1, "This test assumes CONCURRENT_MODELS=1"
    
    client = TonicOllamaClient(debug=True)
    
    # Verify semaphore is initialized with value 1
    assert client.get_available_model_slots() == 1, "Semaphore should have 1 slot available"
    
    # Test semaphore acquisition and release
    async with client._model_semaphore:
        assert client.get_available_model_slots() == 0, "No slots should be available when semaphore is acquired"
    
    assert client.get_available_model_slots() == 1, "Slot should be available after semaphore is released"

@pytest.mark.asyncio
async def test_cannot_override_concurrency_limit():
    """Test that attempting to override the concurrency limit still results in CONCURRENT_MODELS=1."""
    # Try to create a client with a higher concurrency limit
    client = TonicOllamaClient(debug=True, concurrent_models=5)
    
    # Verify semaphore is still initialized with value 1
    assert client.get_available_model_slots() == 1, "Semaphore should have 1 slot regardless of requested concurrent_models"
    assert client.concurrent_models == 1, "concurrent_models should be clamped to 1"
