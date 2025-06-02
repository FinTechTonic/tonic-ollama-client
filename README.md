# Tonic Ollama Client

A Python wrapper for the Ollama API, providing conversation management, model readiness checks, rich console output, and error handling.

## Features

| Feature | Description |
| ------- | ----------- |
| 🔄 **Robust Error Handling** | Automatic retries for connection issues |
| 💬 **Conversation Management** | Track and manage message history |
| ✅ **Model Readiness Checks** | Verify model availability, auto-pull if needed |
| 🎨 **Rich Console Output** | Formatted console output |
| 📊 **Embedding Generation** | Generate text embeddings |
| 🚀 **Support for Key Models** | Optimized for `llama3.1:latest`, `phi4:latest`, `qwen3:8b` |

## Installation

### Using pip

```bash
pip install tonic-ollama-client
```

### Development Setup

```bash
git clone https://github.com/FinTechTonic/tonic-ollama-client
cd tonic-ollama-client
uv pip install -e .[dev]
```

## Basic Usage

### Starting the Ollama Server

Ensure the Ollama server is running:

```bash
ollama serve
```

### Creating a Client

```python
import tonic_ollama_client as toc

client = toc.create_client(debug=True)

# Or with custom settings
client = toc.TonicOllamaClient(
    base_url="http://localhost:11434", 
    debug=True,
    max_readiness_attempts=5
)
```

### Chat with a Model

```python
import asyncio
import tonic_ollama_client as toc

async def chat_example():
    client = toc.create_client(debug=True)
    
    try:
        await client.check_model_ready("llama3.1:latest")
        
        response = await client.chat(
            model="llama3.1:latest",
            message="What is the capital of France?",
            system_prompt="You are a helpful assistant that provides concise answers."
        )
        
        print(response["message"]["content"])
    except toc.OllamaServerNotRunningError:
        print("Please start the Ollama server with 'ollama serve' and try again.")
    except toc.ResponseError as e:
        print(f"Error: {e}")

asyncio.run(chat_example())
```

### Managing Conversations

```python
import asyncio
import tonic_ollama_client as toc

async def conversation_example():
    client = toc.create_client()
    
    conv_id = await client.create_conversation()
    
    response1 = await client.chat(
        model="llama3.1:latest",
        message="Hello, who are you?",
        conversation_id=conv_id
    )
    
    response2 = await client.chat(
        model="llama3.1:latest",
        message="What can you help me with?",
        conversation_id=conv_id
    )
    
    messages = client.get_conversation(conv_id)
    # print(messages) # Example: print messages
    
    client.clear_conversation(conv_id)
    client.delete_conversation(conv_id)

asyncio.run(conversation_example())
```

### Generating Embeddings

```python
import asyncio
import tonic_ollama_client as toc

async def embedding_example():
    client = toc.create_client()
    
    embedding = await client.generate_embedding(
        model="llama3.1:latest",
        text="This is a sample text for embedding generation."
    )
    
    print(f"Generated embedding with {len(embedding)} dimensions")

asyncio.run(embedding_example())
```

## Advanced Features

### Error Handling

```python
import asyncio
import tonic_ollama_client as toc

async def error_handling_example():
    client = toc.create_client()
    
    try:
        await client.check_model_ready("nonexistent-model")
    except toc.OllamaServerNotRunningError:
        print("The Ollama server is not running. Start it with 'ollama serve'")
    except toc.ModelNotReadyError as e:
        print(f"Model not ready: {e.model_name}")
    except toc.ResponseError as e:
        print(f"API error: {e}")
```

## Testing

Run tests with pytest:

```bash
pytest
pytest -s # With console output
OLLAMA_INTEGRATION_TEST=1 pytest -s tests/test_integration.py # Integration tests
```

## License

This project is licensed under the Apache License 2.0 - see LICENSE file for details.
