# Tonic Ollama Client: Basic Usage Example

This document explains the `basic_usage.py` script, which demonstrates the core functionalities of the `tonic-ollama-client`. The script uses `rich` for an enhanced, live-updating console interface and `questionary` for interactive model selection.

## Prerequisites

1.  **Python Environment**: Ensure you have Python 3.11+ installed.
2.  **Tonic Ollama Client & Dependencies**: Install the client library and its dependencies, including `questionary`. If you have cloned the repository, you can install it in editable mode from the project root:
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
4.  **Model Availability**: The script will attempt to list models available on your Ollama server. If it cannot, it will fall back to a default list. Ensure at least one model you intend to use is available (e.g., `ollama pull llama3.1:latest`). The script will warn you if the selected model is not found on the server.

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
# import questionary
# import ollama # For synchronous ollama.list()

# fetched_models = get_ollama_models_sync() # Attempts to get local models
# model_choices = []
# # ... populates choices with fetched or default models
# model_choices.append(CUSTOM_MODEL_OPTION)

# selected_model_or_custom = await questionary.select(...).ask_async()

# if selected_model_or_custom == CUSTOM_MODEL_OPTION:
#     MODEL_NAME = await questionary.text(...).ask_async()
# else:
#     MODEL_NAME = selected_model_or_custom
```
*   It first tries to **dynamically fetch models** currently available on your local Ollama server using `ollama.list()`.
*   If successful, these models are presented in the selection list. Otherwise, a **default list of common models** is shown.
*   You can navigate this list using **up/down arrow keys**.
*   An option **"Enter Custom Model Name..."** is always available. If selected, you'll be prompted to type the model name (e.g., `your-custom-model:tag`).
*   The chosen `MODEL_NAME` is then used for all subsequent operations.

### 1. Client Creation

A `TonicOllamaClient` instance is created:
```python
client = toc.create_client(debug=False) # debug=False for cleaner live display
```
The status log will show the client's base URL and concurrent models limit.

### 2. Ensuring Server Readiness & Model Check

*   **Server Readiness**:
    ```python
    await client.ensure_server_ready()
    ```
    Checks if the Ollama server is responsive, with retries.

*   **Optional Model Availability Check**:
    ```python
    # await client.get_async_client().show(model=MODEL_NAME)
    ```
    Verifies if the *selected* model is available on the server. A warning is shown if not.

### 3. Basic Chat

A simple chat interaction with the *selected* model:
```python
chat_response = await client.chat(
    model=MODEL_NAME,
    message="What is the capital of France? Respond concisely.",
    system_prompt="You are a helpful AI assistant."
)
# ... logs user message and AI response
```

### 4. Conversation Management

Demonstrates multi-turn conversation capabilities with the *selected* model:
*   **Create, Send Messages, Get History, List, Clear, Delete**: Operations are performed similarly to the previous version, but all use the dynamically selected `MODEL_NAME`.

### 5. Embedding Generation

Generates text embeddings using the *selected* model:
```python
embedding = await client.generate_embedding(
    model=MODEL_NAME,
    text="Ollama is a cool tool for running LLMs locally."
)
# ... logs embedding details
```

### 6. Error Handling & Client Closing

*   **Error Handling**: The script robustly handles common errors (`OllamaServerNotRunningError`, `ResponseError`, `ConnectionError`, general `Exception`, `KeyboardInterrupt` during selection), updating the status log within the `Live` display.
*   **Client Closing**: A `finally` block ensures `await client.close()` is called, which attempts to unload predefined models and close the HTTP client. This is also reflected in the progress bar and status log.

## Live Display Interface

The script uses `rich.live.Live` to provide a dynamic, non-scrolling (mostly) interface:
*   **Main Panel**: Shows the script title and the selected model.
*   **Progress Bar**: Displays overall progress through the predefined steps. The description of the current step is updated.
*   **Status Log Panel**: Shows detailed messages for each operation, including outputs from the client (like AI responses) and status updates (e.g., "Ollama server is responsive.").

This interface aims to provide a clear and interactive view of the script's execution flow.
