import asyncio
import tonic_ollama_client as toc
from rich.console import Console, Group
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress as RichProgress, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text
import questionary
from typing import Optional, List
import ollama # For synchronously listing models

# Configuration
DEFAULT_AVAILABLE_MODELS: List[str] = ["llama3.1:latest", "phi4:latest", "qwen2:7b", "mistral:latest"]
CUSTOM_MODEL_OPTION = "Enter Custom Model Name..."

PROGRESS_STEPS = [
    "0. Selecting Model",
    "1. Initializing Client",
    "2. Ensuring Server Readiness",
    "3. Checking Model Availability",
    "4. Performing Basic Chat",
    "5. Conversation: Creating",
    "6. Conversation: Message 1 (Fav Color)",
    "7. Conversation: Message 2 (Query Color)",
    "8. Conversation: Getting History",
    "9. Conversation: Clearing",
    "10. Conversation: Deleting",
    "11. Generating Embeddings",
    "12. Closing Client"
]
TOTAL_STEPS = len(PROGRESS_STEPS)

def get_ollama_models_sync() -> List[str]:
    """Synchronously fetches locally available Ollama models."""
    try:
        models_info = ollama.list()
        if models_info and 'models' in models_info:
            return sorted([model['name'] for model in models_info['models'] if 'name' in model])
    except Exception:
        return []
    return []

async def main():
    console = Console()
    MODEL_NAME: str = ""

    # --- Model Selection Step ---
    console.print(Panel("Welcome to the Tonic Ollama Client Example!", title="[bold green]Setup[/bold green]", expand=False))
    
    fetched_models = get_ollama_models_sync()
    
    model_choices = []
    if fetched_models:
        console.print("\n[bold]Detected Local Ollama Models (use arrow keys to navigate):[/bold]")
        model_choices.extend(fetched_models)
    else:
        console.print("\n[bold]Could not detect local Ollama models. Using default list (use arrow keys to navigate):[/bold]")
        model_choices.extend(DEFAULT_AVAILABLE_MODELS)
    
    model_choices.append(questionary.Separator())
    model_choices.append(CUSTOM_MODEL_OPTION)

    selected_model_or_custom = await questionary.select(
        "Please select a model to use for this session:",
        choices=model_choices,
        use_arrow_keys=True,
        style=questionary.Style([('selected', 'fg:#673ab7 bold'), ('highlighted', 'fg:#673ab7 bold')]),
    ).ask_async()

    if selected_model_or_custom == CUSTOM_MODEL_OPTION:
        MODEL_NAME = await questionary.text(
            "Enter the custom model name (e.g., 'my-model:latest'):"
        ).ask_async()
        if not MODEL_NAME:
            console.print("[bold red]No custom model name entered. Exiting.[/bold red]")
            return
    elif selected_model_or_custom is None:
        console.print("[bold red]No model selected. Exiting.[/bold red]")
        return
    else:
        MODEL_NAME = selected_model_or_custom
    
    console.print(f"\nUsing model: [bold magenta]{MODEL_NAME}[/bold magenta]\n")
    # --- End Model Selection ---

    progress_bar = RichProgress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        expand=True,
    )
    overall_task = progress_bar.add_task(PROGRESS_STEPS[0], total=TOTAL_STEPS, visible=True)
    
    status_display_area = Text("", justify="left") 
    
    live_layout = Group(
        Panel(f"Tonic Ollama Client - Basic Usage Example", 
              title="[bold green]Example Script[/bold green]", expand=False, border_style="green",
              subtitle=f"Selected Model: [bold magenta]{MODEL_NAME}[/bold magenta]"),
        progress_bar,
        status_display_area
    )

    client = None
    
    current_status_lines: List[Text] = []

    with Live(live_layout, console=console, refresh_per_second=10, vertical_overflow="visible") as live:
        
        def _render_status_display():
            new_content = Text()
            for i, line_obj in enumerate(current_status_lines):
                new_content.append_text(line_obj)
                if i < len(current_status_lines) - 1:
                    new_content.append("\n")
            
            status_display_area.truncate(0)
            status_display_area.append_text(new_content)

        def update_status(message: str, style: str = "white", step_index: Optional[int] = None, advance: bool = True):
            nonlocal overall_task
            if step_index is not None:
                current_status_lines.clear() 
                current_description = PROGRESS_STEPS[step_index]
                if advance:
                    progress_bar.update(overall_task, description=current_description, advance=1)
                else:
                    progress_bar.update(overall_task, description=current_description)
                current_status_lines.append(Text.from_markup(f"[bold cyan]{current_description}[/bold cyan]"))
            
            if step_index is None and len(current_status_lines) > 1:
                current_status_lines[-1] = Text(f"   {message}", style=style)
            else:
                current_status_lines.append(Text(f"   {message}", style=style))

            _render_status_display()
            live.refresh()

        def log_output(label: str, content: str, style: str = "dim white", max_content_len: int = 70):
            if len(content) > max_content_len:
                content = content[:max_content_len-3] + "..."
            current_status_lines.append(Text.from_markup(f"     [bold]{label}:[/bold] {content}", style=style))
            _render_status_display()
            live.refresh()

        try:
            update_status(f"Selected Model: [magenta]{MODEL_NAME}[/magenta]", step_index=0, advance=True)

            update_status("Creating client instance...", step_index=1)
            client = toc.create_client(debug=False)
            update_status(f"Client created (URL: {client.base_url}).", style="green", advance=False)

            update_status("Ensuring Ollama server is responsive...", step_index=2)
            await client.ensure_server_ready()
            update_status("Ollama server is responsive.", style="green", advance=False)

            update_status(f"Checking availability of model '{MODEL_NAME}'...", step_index=3)
            try:
                await client.get_async_client().show(model=MODEL_NAME)
                update_status(f"Model '{MODEL_NAME}' is available.", style="green", advance=False)
            except toc.ResponseError as e:
                if e.status_code == 404:
                    update_status(f"Warning: Model '{MODEL_NAME}' not found.", style="yellow", advance=False)
                    log_output("Suggestion", f"Run: ollama pull {MODEL_NAME}", style="dim yellow")
                else:
                    raise
            
            update_status(f"Performing basic chat with '{MODEL_NAME}'...", step_index=4)
            user_msg_basic = "What is the capital of France? Respond concisely."
            chat_response = await client.chat(
                model=MODEL_NAME,
                message=user_msg_basic,
                system_prompt="You are a helpful AI assistant."
            )
            log_output("User", user_msg_basic)
            log_output("AI", chat_response['message']['content'])

            update_status("Creating conversation 'my-live-test-conversation'...", step_index=5)
            conv_id = await client.create_conversation("my-live-test-conversation")
            log_output("Conv ID", conv_id)

            update_status(f"Sending message to '{conv_id}'...", step_index=6)
            user_msg_conv1 = "Hello! My favorite color is blue."
            await client.chat(
                model=MODEL_NAME,
                message=user_msg_conv1,
                conversation_id=conv_id,
                system_prompt="Remember details about the user."
            )
            log_output("User", user_msg_conv1)
            
            update_status(f"Querying context in '{conv_id}'...", step_index=7)
            user_msg_conv2 = "What is my favorite color?"
            response_conv = await client.chat(
                model=MODEL_NAME,
                message=user_msg_conv2,
                conversation_id=conv_id
            )
            log_output("User", user_msg_conv2)
            log_output("AI", response_conv['message']['content'])

            update_status(f"Retrieving history for '{conv_id}'...", step_index=8)
            history = client.get_conversation(conv_id)
            log_output("History items", str(len(history)))

            update_status(f"Clearing conversation '{conv_id}'...", step_index=9)
            client.clear_conversation(conv_id)
            log_output("History after clear", str(len(client.get_conversation(conv_id))))

            update_status(f"Deleting conversation '{conv_id}'...", step_index=10)
            client.delete_conversation(conv_id)
            log_output("Exists after delete?", str(conv_id in client.list_conversations()))

            update_status(f"Generating embeddings with '{MODEL_NAME}'...", step_index=11)
            embedding_text = "Ollama is a cool tool for running LLMs locally."
            embedding = await client.generate_embedding(
                model=MODEL_NAME,
                text=embedding_text
            )
            log_output("Embedding Dims", str(len(embedding)))
            update_status(f"Embeddings generated ({len(embedding)} dims).", style="green", advance=False)

            update_status("Example script completed successfully!", style="bold green", advance=False)

        except toc.OllamaServerNotRunningError as e:
            update_status(f"Ollama Server Error: {e}", style="bold red", advance=False)
            log_output("Action", "Please ensure the Ollama server is running ('ollama serve') and accessible.", style="red")
        except toc.ResponseError as e:
            update_status(f"Ollama API Response Error (Status {e.status_code}): {e.error}", style="bold red", advance=False)
        except ConnectionError as e:
            update_status(f"Connection Error: {e}", style="bold red", advance=False)
            log_output("Details", "Could not connect to the Ollama server. Check network and server status.", style="red")
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                update_status("Model selection or input cancelled by user.", style="bold yellow", advance=False)
            else:
                update_status(f"An unexpected error occurred: {type(e).__name__}: {e}", style="bold red", advance=False)
        finally:
            current_progress = progress_bar.tasks[0].completed if progress_bar.tasks else 0
            is_last_step_pending = current_progress < TOTAL_STEPS

            if client:
                update_status("Cleaning up: Closing client...", style="blue", 
                              step_index=12 if is_last_step_pending else None, 
                              advance=is_last_step_pending)
                await client.close()
                current_status_lines.append(Text("   Client closed.", style="blue"))
                _render_status_display()
                live.refresh()
            else:
                 if is_last_step_pending and current_progress < 12 :
                    progress_bar.update(overall_task, description=PROGRESS_STEPS[12], advance=1)
                 current_status_lines.append(Text("   Client not initialized or cleanup skipped.", style="dim blue"))
                 _render_status_display()
                 live.refresh()
            
            final_completed_steps = progress_bar.tasks[0].completed if progress_bar.tasks else 0
            if final_completed_steps < TOTAL_STEPS:
                 progress_bar.update(overall_task, completed=TOTAL_STEPS, description="Finished.")
            elif progress_bar.tasks:
                 progress_bar.update(overall_task, description="Finished.")
            
            current_status_lines.append(Text("\n[bold blue]End of example.[/bold blue]"))
            _render_status_display()
            live.refresh()

if __name__ == "__main__":
    asyncio.run(main())
