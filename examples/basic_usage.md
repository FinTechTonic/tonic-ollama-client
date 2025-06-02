# Tonic Ollama Client: Basic Usage Example

This document explains the `basic_usage.py` script, which demonstrates the core functionalities of the `tonic-ollama-client`.

## Prerequisites

1.  **Python Environment**: Ensure you have Python 3.11+ installed.
2.  **Tonic Ollama Client**: Install the client library. If you have cloned the repository, you can install it in editable mode from the project root:
    ```bash
    # Make sure you are in the root of the tonic-ollama-client repository
    pip install -e .
    # Or, if you have 'uv' installed:
    # uv pip install -e .
    ```
    If you are installing it as a published package (once available):
    ```bash
    pip install tonic-ollama-client
    ```
3.  **Ollama Server**: The Ollama server must be running. You can typically start it with:
    ```bash
    ollama serve
    ```
4.  **Model Availability**: The example script uses a model specified by the `MODEL_NAME` variable (e.g., `llama3.1:latest`). Ensure this model is available on your Ollama server. If not, pull it:
    ```bash
    ollama pull llama3.1:latest
    # or ollama pull phi4:latest, ollama pull qwen2:7b, etc.
    # The script will warn you if the model is not found.
    ```

## Running the Example

Navigate to the `examples` directory (relative to the project root) and run the script:

```bash
python basic_usage.py
```

## Script Breakdown

The `basic_usage.py` script showcases the following features using `rich` for enhanced console output:

### 1. Client Creation

A `TonicOllamaClient` instance is created using the `toc.create_client()` factory function:
```python
client = toc.create_client(debug=True)
```
Setting `debug=True` enables verbose console output, showing client operations, internal checks, and semaphore usage. The script will print the client's base URL, debug mode status, and the configured concurrent models limit (which is hardcoded to 1 in the current client version).

### 2. Ensuring Server Readiness & Model Check

*   **Server Readiness**: Before making API calls, the script ensures the Ollama server is responsive:
    ```python
    await client.ensure_server_ready()
    ```
    This method checks if the Ollama server at the configured `base_url` is running and accessible. It will retry a few times if the server is not immediately responsive.

*   **Optional Model Availability Check**: The script then performs an explicit check to see if the target `MODEL_NAME` is available on the server using the underlying `ollama.AsyncClient.show()` method:
    ```python
    try:
        await client.get_async_client().show(model=MODEL_NAME)
        # ... model is available
    except toc.ResponseError as e:
        if e.status_code == 404:
            # ... model not found, print warning
    ```
    This is a good practice as `ensure_server_ready()` only checks for server responsiveness, not specific model availability.

### 3. Basic Chat

A simple chat interaction is demonstrated:
```python
chat_response = await client.chat(
    model=MODEL_NAME,
    message="What is the capital of France? Respond concisely.",
    system_prompt="You are a helpful AI assistant."
)
console.print(f"   [bold]AI ({MODEL_NAME}):[/bold] {chat_response['message']['content']}")
```
This sends a message to the specified model with a system prompt and prints the AI's response.

### 4. Conversation Management

The script demonstrates how to manage multi-turn conversations:

*   **Create a Conversation**:
    ```python
    conv_id = await client.create_conversation("my-test-conversation")
    ```
    A unique ID is assigned to the conversation. If no ID is provided, a UUID is generated.

*   **Send Messages within a Conversation**:
    Subsequent calls to `client.chat()` using the same `conversation_id` will maintain context. The client automatically appends user and assistant messages to the history for that `conversation_id`.
    ```python
    await client.chat(
        model=MODEL_NAME,
        message="Hello! My favorite color is blue.",
        conversation_id=conv_id,
        system_prompt="Remember details about the user."
    )
    # ...
    response_conv = await client.chat(
        model=MODEL_NAME,
        message="What is my favorite color?",
        conversation_id=conv_id
    )
    ```

*   **Get Conversation History**:
    ```python
    history = client.get_conversation(conv_id)
    ```
    Retrieves all messages (user and assistant) for a given conversation.

*   **List Conversations**:
    ```python
    assert "my-test-conversation" in client.list_conversations()
    ```
    Returns a list of all active conversation IDs.

*   **Clear a Conversation**:
    ```python
    client.clear_conversation(conv_id)
    ```
    Removes all messages from a conversation but keeps the conversation ID (the history list becomes empty).

*   **Delete a Conversation**:
    ```python
    client.delete_conversation(conv_id)
    ```
    Removes the conversation and its history entirely from the client's management.

### 5. Embedding Generation

The script shows how to generate text embeddings:
```python
embedding = await client.generate_embedding(
    model=MODEL_NAME,
    text="Ollama is a cool tool for running LLMs locally."
)
console.print(f"   - Embedding dimensions: {len(embedding)}")
```
This generates a vector representation (list of floats) of the input text using the specified model.

### 6. Error Handling

The entire `main` function is wrapped in a `try...except...finally` block to catch and report common errors:
*   `toc.OllamaServerNotRunningError`: If the Ollama server is not accessible after retry attempts.
*   `toc.ResponseError`: For API-specific errors returned by the Ollama server (e.g., model not found (404), malformed requests). The status code and error message are printed.
*   `ConnectionError`: For general network connection issues.
*   `Exception`: A catch-all for any other unexpected errors, printing the type and message.
The `finally` block ensures a concluding message is always printed.

## Expected Output

When run successfully, the script will print detailed, color-coded information about each step using the `rich` library:
*   Client initialization details.
*   Server and model readiness status.
*   User prompts and AI responses for the basic chat.
*   Actions taken during conversation management (creation, messages sent, history retrieval, clearing, deletion).
*   Confirmation of embedding generation and a snippet of the embedding.
*   A success message upon completion.

If errors occur (e.g., server not running, model not found), informative error messages will be displayed in distinct colors. The `debug=True` setting for the client will also produce additional logs from the `TonicOllamaClient` regarding its internal operations, such as semaphore acquisition/release for model concurrency.
