import asyncio
import tonic_ollama_client as toc
from rich.console import Console
from rich.panel import Panel

# Configuration
MODEL_NAME = "llama3.1:latest"  # Or "phi4:latest", "qwen2:7b"
# MODEL_NAME = "phi3:latest" # Example of a model that might not be in APPROVED_MODELS_TO_CLEANUP

async def main():
    console = Console()
    console.print(Panel(f"Tonic Ollama Client - Basic Usage Example (Model: {MODEL_NAME})", 
                        title="[bold green]Example Script[/bold green]", expand=False))

    # 1. Create a client instance
    # debug=True provides more verbose output about client operations
    client = toc.create_client(debug=True)
    console.print("\n[bold cyan]1. Client Created[/bold cyan]")
    console.print(f"   - Base URL: {client.base_url}")
    console.print(f"   - Debug Mode: {client.debug}")
    console.print(f"   - Concurrent Models Limit: {client.concurrent_models}")

    try:
        # 2. Ensure Ollama server is ready (and optionally the model, though ensure_server_ready only checks server)
        console.print("\n[bold cyan]2. Ensuring Server Readiness...[/bold cyan]")
        await client.ensure_server_ready()
        console.print("   [green]Ollama server is responsive.[/green]")

        # Optional: A more explicit check if a specific model is available
        # This is good practice before making specific model calls if ensure_server_ready doesn't check models.
        try:
            await client.get_async_client().show(model=MODEL_NAME)
            console.print(f"   [green]Model '{MODEL_NAME}' is available on the server.[/green]")
        except toc.ResponseError as e:
            if e.status_code == 404:
                console.print(f"   [bold yellow]Warning: Model '{MODEL_NAME}' not found on Ollama server. Chat/embedding may fail or trigger a pull.[/bold yellow]")
                console.print(f"   [dim]You might need to run: ollama pull {MODEL_NAME}[/dim]")
            else:
                raise # Re-raise other ResponseErrors

        # 3. Basic Chat Example
        console.print(f"\n[bold cyan]3. Basic Chat with '{MODEL_NAME}'[/bold cyan]")
        chat_response = await client.chat(
            model=MODEL_NAME,
            message="What is the capital of France? Respond concisely.",
            system_prompt="You are a helpful AI assistant."
        )
        console.print(f"   [bold]User:[/bold] What is the capital of France? Respond concisely.")
        console.print(f"   [bold]AI ({MODEL_NAME}):[/bold] {chat_response['message']['content']}")

        # 4. Conversation Management Example
        console.print("\n[bold cyan]4. Conversation Management[/bold cyan]")
        
        # Create a new conversation
        conv_id = await client.create_conversation("my-test-conversation")
        console.print(f"   - Created conversation with ID: {conv_id}")

        # First message in the conversation
        await client.chat(
            model=MODEL_NAME,
            message="Hello! My favorite color is blue.",
            conversation_id=conv_id,
            system_prompt="Remember details about the user."
        )
        console.print(f"   - Sent to '{conv_id}': Hello! My favorite color is blue.")

        # Second message, AI should ideally remember the context
        response_conv = await client.chat(
            model=MODEL_NAME,
            message="What is my favorite color?",
            conversation_id=conv_id
        )
        console.print(f"   - Sent to '{conv_id}': What is my favorite color?")
        console.print(f"   - AI's response in '{conv_id}': {response_conv['message']['content']}")

        # Get conversation history
        history = client.get_conversation(conv_id)
        console.print(f"   - Conversation history for '{conv_id}' ({len(history)} messages):")
        for msg in history:
            console.print(f"     - {msg['role']}: {msg['content'][:70]}...") # Print snippet

        # Clear conversation
        client.clear_conversation(conv_id)
        console.print(f"   - Cleared conversation: {conv_id}")

        # Delete conversation
        client.delete_conversation(conv_id)
        console.print(f"   - Deleted conversation: {conv_id}")

        # 5. Embedding Generation Example
        console.print(f"\n[bold cyan]5. Generating Embeddings with '{MODEL_NAME}'[/bold cyan]")
        embedding_text = "Ollama is a cool tool for running LLMs locally."
        embedding = await client.generate_embedding(
            model=MODEL_NAME,
            text=embedding_text
        )
        console.print(f"   - Generated embedding for: \"{embedding_text}\"")
        console.print(f"   - Embedding dimensions: {len(embedding)}")
        console.print(f"   - Embedding (first 5 values): {embedding[:5]}...")

        console.print("\n[bold green]Example script completed successfully![/bold green]")

    except toc.OllamaServerNotRunningError as e:
        console.print(f"\n[bold red]Ollama Server Error:[/bold red]\n{e}")
        console.print("Please ensure the Ollama server is running ('ollama serve') and accessible.")
    except toc.ResponseError as e:
        console.print(f"\n[bold red]Ollama API Response Error:[/bold red]\nStatus Code: {e.status_code}\nError: {e.error}")
    except ConnectionError as e:
        console.print(f"\n[bold red]Connection Error:[/bold red]\n{e}")
        console.print("Could not connect to the Ollama server. Check network and server status.")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred:[/bold red]\n{type(e).__name__}: {e}")
    finally:
        if 'client' in locals() and client is not None: # Check if client was initialized
            console.print("\n[bold blue]Cleaning up: Closing client...[/bold blue]")
            await client.close()
        console.print("\n[bold blue]End of example.[/bold blue]")


if __name__ == "__main__":
    asyncio.run(main())
