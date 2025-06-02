# Tonic Ollama Client: Autonomous Conversation Demo

This document explains the `convo_simple.py` script, which demonstrates how to create a simulated conversation between two AI personas using the `tonic-ollama-client` library.

## Overview

The script allows users to:
1. Select two distinct personas from a list of famous computer scientists
2. Choose or create a conversation starter
3. Set a maximum number of conversation turns
4. Watch an autonomous conversation unfold between the selected personas

Each persona has a unique system prompt that guides their behavior and response style during the conversation.

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.11+ installed.
2. **Tonic Ollama Client & Dependencies**: Install the client library and dependencies:
   ```bash
   pip install -e .
   ```
3. **Ollama Server**: Make sure the Ollama server is running:
   ```bash
   ollama serve
   ```
4. **Model Availability**: The script uses `mistral:latest` by default. Ensure it's available:
   ```bash
   ollama pull mistral:latest
   ```

## Running the Example

Navigate to the `examples` directory and run:

```bash
python convo_simple.py
```

## Interactive Setup Process

The script guides you through a series of selections:

### 1. Persona Selection

First, select the first persona from a list of famous computer scientists:
- Alan Turing
- Grace Hopper
- John von Neumann
- Ada Lovelace
- Claude Shannon

Then, select the second persona (the first selection will be excluded).

### 2. Conversation Starter

Choose a conversation starter from the preset options or create your own custom starter.

### 3. Conversation Length

Specify the maximum number of conversation turns:
- Enter `-1` for an infinite conversation (can be stopped with Ctrl+C)
- Enter any number greater than or equal to `2` for a fixed number of turns

## Conversation Process

After setup, the conversation begins:
1. The first persona initiates with the selected conversation starter
2. The second persona responds
3. The conversation continues back and forth until the maximum number of turns is reached or the user interrupts with Ctrl+C

## Technical Implementation

The script demonstrates several features of the Tonic Ollama Client:

### 1. Client Initialization and Server Readiness

```python
client = toc.create_client(debug=False)
await client.ensure_server_ready()
```

### 2. Model Availability Check

```python
await client.get_async_client().show(model=DEFAULT_MODEL)
```

### 3. Conversation Management

Each persona gets its own conversation history with a unique system prompt:

```python
persona1_history = [{"role": "system", "content": PERSONAS[persona1]}]
persona2_history = [{"role": "system", "content": PERSONAS[persona2]}]
```

### 4. Interactive Chat Loop

The script manages a back-and-forth conversation between the two personas:

```python
response = await client.chat(
    model=DEFAULT_MODEL,
    message=current_prompt,
    temperature=0.8,
    conversation_id=f"persona_{current_speaker.replace(' ', '_').lower()}"
)
```

### 5. Rich Interactive Console

The script uses `rich` to create a dynamic, live-updating display showing:
- Conversation progress
- Current turn indicator
- Color-coded messages for each persona

### 6. Clean Termination

The script properly closes the client, unloading the used model:

```python
await client.close(model_to_unload=DEFAULT_MODEL)
```

## Customization

You can modify the script to:
- Add new personas by extending the `PERSONAS` dictionary
- Change the default model by modifying `DEFAULT_MODEL`
- Add new conversation starters to `CONVERSATION_STARTERS`
- Adjust the conversation appearance by modifying the Rich display elements

## Handling Different Conversation Styles

The conversation in this example is turn-based with each persona responding directly to the previous message. This approach keeps the implementation simple while demonstrating the key concepts.

For more sophisticated conversation patterns, you could:
1. Implement memory or summarization for longer contexts
2. Allow personas to ask questions to each other
3. Add a moderator or specific topics to guide the conversation

## Error Handling

The script includes robust error handling for:
- Ollama server not running
- Model not available
- User interruption via Ctrl+C
- API errors during conversation
