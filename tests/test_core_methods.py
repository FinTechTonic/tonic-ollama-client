import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
import asyncio
from tonic_ollama_client import TonicOllamaClient, create_client, get_ollama_models_sync

@pytest.mark.asyncio
async def test_create_client_factory():
    """Test the create_client factory function."""
    # Basic client creation
    client1 = create_client(debug=True)
    assert isinstance(client1, TonicOllamaClient)
    assert client1.debug is True
    
    # Custom settings
    client2 = create_client(
        base_url="http://test:1234",
        max_server_startup_attempts=5,
        debug=False,
        stream_responses=True
    )
    assert client2.base_url == "http://test:1234"
    assert client2.max_server_startup_attempts == 5
    assert client2.debug is False
    assert client2.stream_responses is True

@pytest.mark.asyncio
async def test_get_ollama_models_sync_function():
    """Test the get_ollama_models_sync function."""
    with patch('ollama.list') as mock_list:
        # Mock a successful response
        mock_list.return_value = {
            'models': [
                {'name': 'model1:latest'},
                {'name': 'model2:latest'}
            ]
        }
        
        models = get_ollama_models_sync()
        assert models == ['model1:latest', 'model2:latest']
        
        # Mock a failure
        mock_list.side_effect = Exception("Test error")
        models = get_ollama_models_sync()
        assert models == []

@pytest.mark.asyncio
async def test_close_method_with_model():
    """Test the close method with a specific model."""
    with patch('tonic_ollama_client.AsyncClient') as MockAsyncClient:
        mock_instance = MockAsyncClient.return_value
        mock_instance.generate = AsyncMock()
        mock_instance._client = AsyncMock()
        mock_instance._client.aclose = AsyncMock()
        
        client = TonicOllamaClient(debug=True)
        # Force client to get the async client instance
        client.get_async_client()
        
        # Call close with a specific model
        await client.close(model_to_unload="test-model")
        
        # Verify generate was called with the right parameters
        mock_instance.generate.assert_called_once_with(
            model="test-model",
            prompt=".",
            options={"num_predict": 1},
            keep_alive="0s"
        )
        
        # Verify the underlying client was closed
        mock_instance._client.aclose.assert_called_once()
        
        # Verify the client was cleared
        assert client.async_client is None

@pytest.mark.asyncio
async def test_close_method_with_default_models():
    """Test the close method with default models."""
    with patch('tonic_ollama_client.AsyncClient') as MockAsyncClient:
        mock_instance = MockAsyncClient.return_value
        mock_instance.generate = AsyncMock()
        mock_instance._client = AsyncMock()
        mock_instance._client.aclose = AsyncMock()
        
        # Create a client with custom models to unload
        test_models = ["custom1", "custom2"]
        client = TonicOllamaClient(debug=True, models_to_unload_on_close=test_models)
        # Force client to get the async client instance
        client.get_async_client()
        
        # Call close without a specific model (should use default list)
        await client.close()
        
        # Verify generate was called for each model
        assert mock_instance.generate.call_count == len(test_models)
        for model in test_models:
            mock_instance.generate.assert_any_call(
                model=model,
                prompt=".",
                options={"num_predict": 1},
                keep_alive="0s"
            )
        
        # Verify the underlying client was closed
        mock_instance._client.aclose.assert_called_once()
        
        # Verify the client was cleared
        assert client.async_client is None
