# Tonic Ollama Client: Basic Usage Example

This document explains the `basic_usage.py` script, which demonstrates the core functionalities of the `tonic-ollama-client`. The script uses `rich` for an enhanced, live-updating console interface and `questionary` for interactive model selection.

## Prerequisites

1.  **Python Environment**: Ensure you have Python 3.11+ installed.
2.  **Tonic Ollama Client & Dependencies**: Install the client library and its dependencies. If you have cloned the repository, you can install it in editable mode from the project root:
    ```bash
    # Make sure you are in the root of the tonic-ollama-client repository
    pip install -e .
    # Or, if you have 'uv' installed:
    # uv pip install -e .
    ```
    This will install `tonic-ollama-client` along with `ollama`, `rich`, `tenacity`, and `questionary` as specified in `pyproject.toml`.
3.  **Ollama Server**: The Ollama server must be running. You can typically start it with:
    ```bash
    ollama serve
    ```
4.  **Model Availability**: The script will attempt to list models available on your Ollama server using `toc.get_ollama_models_sync()`. If it cannot detect any models, it will fall back to a default list. Ensure at least one model you intend to use is available (e.g., `ollama pull llama3.1:latest`). The script will warn you if the selected model is not found on the server.

## Running the Example

Navigate to the `examples` directory (relative to the project root) and run the script:

```bash
python basic_usage.py
```
You will first be prompted to select a model from an interactive list using arrow keys, or you can choose to enter a custom model name.

## Script Breakdown

The `basic_usage.py` script showcases the following features using a `rich.live.Live` display for a dynamic console output, including a progress bar and a status log.

### 0. Interactive Model Selection (Enhanced)

Before any operations begin, the script provides an interactive model selection menu:
```python
# Fetch models using the client's built-in function
fetched_models = toc.get_ollama_models_sync()

# Build choice list with detected or default models
model_choices = []
if fetched_models:
    model_choices.extend(fetched_models)
else:
    model_choices.extend(DEFAULT_AVAILABLE_MODELS)

# Always include custom model option
model_choices.append(questionary.Separator())
model_choices.append(CUSTOM_MODEL_OPTION)

# Interactive selection
selected_model_or_custom = await questionary.select(
    "Please select a model to use for this session:",
    choices=model_choices,
    use_arrow_keys=True
).ask_async()
```

**Features:**
*   **Dynamic Model Detection**: Uses `toc.get_ollama_models_sync()` to fetch models currently available on your local Ollama server
*   **Fallback to Defaults**: If model detection fails, shows a predefined list of common models
*   **Arrow Key Navigation**: Navigate the selection list using up/down arrow keys
*   **Custom Model Entry**: Option to enter any custom model name (e.g., `your-custom-model:tag`)
*   **Graceful Exit**: Handles user cancellation (Esc key) appropriately

### 1. Client Creation

A `TonicOllamaClient` instance is created with debug output disabled for cleaner live display:
```python
client = toc.create_client(debug=False)
```
The status display shows the client's base URL upon successful creation.

### 2. Server Readiness & Model Availability Check

**Server Readiness:**
```python
await client.ensure_server_ready()
```
Checks if the Ollama server is responsive with configurable retry attempts.

**Model Availability Check:**
```python
await client.get_async_client().show(model=MODEL_NAME)
```
Verifies if the selected model is available on the server. Shows a warning with pull suggestion if the model is not found (404 error).

### 3. Basic Chat Interaction

Demonstrates a simple chat interaction with the selected model:
```python
chat_response = await client.chat(
    model=MODEL_NAME,
    message="What is the capital of France? Respond concisely.",
    system_prompt="You are a helpful AI assistant."
)
```
Logs both the user message and AI response in the live status display.

### 4. Advanced Conversation Management

Shows multi-turn conversation capabilities with context retention:

**Create Conversation:**
```python
conv_id = await client.create_conversation("my-live-test-conversation")
```

**Multi-turn Chat with Context:**
```python
# First message with system prompt
await client.chat(
    model=MODEL_NAME,
    message="Hello! My favorite color is blue.",
    conversation_id=conv_id,
    system_prompt="Remember details about the user. Be friendly and conversational."
)

# Follow-up messages that reference previous context
await client.chat(
    model=MODEL_NAME,
    message="What is my favorite color?",
    conversation_id=conv_id
)

# Additional contextual conversations
await client.chat(
    model=MODEL_NAME,
    message="Thanks! Based on my favorite color, can you suggest a type of flower I might like?",
    conversation_id=conv_id
)
```

**Conversation Operations:**
- **History Retrieval**: `client.get_conversation(conv_id)`
- **Clear Messages**: `client.clear_conversation(conv_id)`  
- **Delete Conversation**: `client.delete_conversation(conv_id)`

### 5. Embedding Generation

Generates text embeddings using the selected model:
```python
embedding = await client.generate_embedding(
    model=MODEL_NAME,
    text="Ollama is a cool tool for running LLMs locally."
)
```
Displays embedding dimensions and confirms successful generation.

### 6. Comprehensive Error Handling

The script handles various error scenarios:

**Server Errors:**
```python
except toc.OllamaServerNotRunningError as e:
    update_status(f"Ollama Server Error: {e}", style="bold red")
```

**API Response Errors:**
```python
except toc.ResponseError as e:
    update_status(f"Ollama API Response Error (Status {e.status_code}): {e.error}", style="bold red")
```

**Connection Issues:**
```python
except ConnectionError as e:
    update_status(f"Connection Error: {e}", style="bold red")
```

**User Cancellation:**
```python
except Exception as e:
    if isinstance(e, KeyboardInterrupt):
        update_status("Model selection or input cancelled by user.", style="bold yellow")
```

### 7. Model-Specific Client Cleanup

Enhanced cleanup with model-specific unloading:
```python
# In the finally block
await client.close(model_to_unload=MODEL_NAME)
```

**Features:**
- **Targeted Unloading**: Unloads only the specific model used in the session
- **Graceful Cleanup**: Properly closes HTTP connections and releases resources
- **Progress Tracking**: Updates progress bar and status display during cleanup

## Live Display Interface

The script uses `rich.live.Live` to provide a dynamic, non-scrolling interface:

**Components:**
- **Header Panel**: Shows script title and selected model name
- **Progress Bar**: 13-step progress indicator with current step description and completion percentage
- **Dynamic Status Area**: Shows current operation details, user-AI interactions, and status messages

**Key Features:**
- **Non-scrolling Updates**: Status area content is replaced (not appended) for each step
- **Rich Formatting**: Color-coded messages (green for success, yellow for warnings, red for errors)
- **Real-time Progress**: Progress bar updates as operations complete
- **Concise Output**: Long content is truncated to maintain clean display

## Supported Models

The script works with any Ollama-compatible model, with built-in support for:
- `llama3.1:latest`
- `phi4:latest` 
- `qwen3:8b`
- `mistral:latest`

Custom models can be entered manually during the selection process.

## Error Recovery

The script includes robust error recovery:
- **Automatic Retries**: Uses tenacity for transient connection issues
- **Graceful Degradation**: Continues execution when possible after non-fatal errors
- **Clear Error Messages**: Provides actionable feedback for common issues
- **Resource Cleanup**: Ensures proper cleanup even when errors occur
