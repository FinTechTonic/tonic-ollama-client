import asyncio
import datetime
import os
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

import questionary
import tiktoken
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, TextColumn,
                           TimeRemainingColumn)
from rich.text import Text

import tonic_ollama_client as toc

# Configuration
DEFAULT_ENCODING = "cl100k_base"
REPROMPT_TOKEN_THRESHOLD = 200  # Remind persona after this many generated tokens

# System prompt template
GENERIC_SYSTEM_PROMPT_TEMPLATE = """
You are {persona_name}. Embody this historical figure completely. Your thoughts, reasoning, and responses must perfectly reflect their knowledge, personality, and communication style.

**Persona Profile:**
{persona_description}

**Conversation Directives (Strict Adherence Required):**
1.  **Unyielding Authenticity:** Consistently channel {persona_name}'s distinct voice, perspective, and mannerisms in every interaction.
2.  **Historical Foundation:** Ground all statements and reasoning in {persona_name}'s actual historical knowledge, expertise, and worldview.
3.  **Engaged Dialogue:** Respond thoughtfully and directly to the other speaker's points, fostering a meaningful exchange.
4.  **Seek Clarity (In Character):** If any part of the conversation is unclear, politely request clarification as {persona_name} would.
5.  **Proactive Contribution:** Actively ask questions and express opinions that are authentically aligned with {persona_name}'s known views and intellectual curiosities.
6.  **Natural Conversational Flow:** Maintain a fluid and natural conversational rhythm. Introduce related topics that would genuinely interest {persona_name}.
7.  **Optimal Response Length:** Aim for responses of 1-3 paragraphs to encourage a dynamic, back-and-forth dialogue.
8.  **Authentic Lexicon:** Employ vocabulary, references, and illustrative examples that are characteristic of {persona_name} and their era.
9.  **Guiding Peer's Persona:** Should the other speaker deviate from their assumed persona, gently and in-character, guide them back or pose a question to encourage their return to character.

Remember, you are conversing with another historical figure. Your primary directive is to engage with their ideas with the full authenticity of {persona_name}—whether that involves respectful agreement, passionate debate, or insightful inquiry. Strive for a deeply authentic and intellectually stimulating dialogue.
"""

PERSONAS = {
    "Alan Turing": "You are Alan Turing, a brilliant mathematician and computer scientist known for breaking the Enigma code during WWII and pioneering work in computing. You are thoughtful, analytical, and interested in artificial intelligence, having proposed the Turing Test.",
    "Ada Lovelace": "You are Ada Lovelace, the world's first computer programmer who worked on Charles Babbage's Analytical Engine. You combine mathematical logic with poetic imagination and foresee computers going beyond mere calculation.",
    "Albert Einstein": "You are Albert Einstein, the revolutionary physicist who developed the theory of relativity. You approach problems with thought experiments and visual imagination. You're philosophical, pacifistic, and believe in simplicity. You often use analogies and thought experiments to explain complex ideas.",
    "Charles Darwin": "You are Charles Darwin, the naturalist who proposed the theory of evolution by natural selection. You are methodical, observant, and cautious. You collect extensive evidence before drawing conclusions and are willing to challenge established beliefs when evidence demands it.",
    "Aristotle": "You are Aristotle, the ancient Greek philosopher who made foundational contributions to logic, metaphysics, ethics, and natural sciences. You approach knowledge systematically, categorizing, and analyzing concepts. You believe in empirical observation and logical reasoning as paths to understanding."
}

# Conversation starters
CONVERSATION_STARTERS = [
    "Will AI Reveloutionize human society, or will it lead to dystopia? What are the most likely scenarios to occur and when?",
    "Are religious beliefs fundamentally incompatible with scientific thinking? Don't hold back on your true views.",
    "Free will is clearly an illusion given what we know about physics and neuroscience. Would you agree, and what implications does this have for human responsibility?",
    "The scientific establishment routinely suppresses radical ideas. Was your work constrained by orthodoxy, or did you perpetuate it?",
    "Has academic credentialism and elitism damaged scientific progress? Did your own success depend on privilege rather than merit?",
    "Is human-level AI inevitable, and if so, will it render human intellect obsolete? Should we fear this outcome?",
    "Custom starter (type your own)..."
]

# Model options
DEFAULT_AVAILABLE_MODELS = ["llama3.1:latest", "phi4:latest", "qwen3:8b", "mistral:latest"]
CUSTOM_MODEL_OPTION = "Enter Custom Model Name..."
PROGRESS_STEPS = [f"Step {i}" for i in range(20)]


def clear_console():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_timestamp():
    """Get current timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Count tokens in text."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4  # Rough approximation

def count_message_tokens(message: Dict[str, str], encoding_name: str = DEFAULT_ENCODING) -> int:
    """Count tokens in a message."""
    total = 0
    total += count_tokens(message.get("role", ""), encoding_name)
    if "content" in message and message["content"]:
        total += count_tokens(message["content"], encoding_name)
    return total

def log_conversation_to_file(persona1: str, persona2: str, conversation_history: List[Tuple[str, str]], model_name: str):
    """Save conversation to file."""
    timestamp = get_timestamp()
    filename = f"conversation_{persona1.replace(' ', '_')}_{persona2.replace(' ', '_')}_{timestamp}.log"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Conversation between {persona1} and {persona2}\n")
        f.write(f"Using model: {model_name}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for speaker, message in conversation_history:
            f.write(f"{speaker}:\n{message}\n\n")
    
    return filename

def get_full_system_prompt(persona_name: str) -> str:
    """Generate system prompt for persona."""
    persona_description = PERSONAS[persona_name]
    return GENERIC_SYSTEM_PROMPT_TEMPLATE.format(
        persona_description=persona_description,
        persona_name=persona_name
    )

async def select_model(console: Console) -> str:
    """Select LLM model."""
    console.print(Panel(
        "Select an LLM model to power the conversation",
        title="[bold blue]Model Selection[/bold blue]",
        border_style="blue"
    ))
    
    fetched_models = toc.get_ollama_models_sync()
    
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
        "Please select a model to use for this conversation:",
        choices=model_choices,
        use_arrow_keys=True,
        style=questionary.Style([('selected', 'fg:#673ab7 bold'), ('highlighted', 'fg:#673ab7 bold')]),
    ).ask_async()

    if selected_model_or_custom == CUSTOM_MODEL_OPTION:
        model_name = await questionary.text(
            "Enter the custom model name (e.g., 'my-model:latest'):"
        ).ask_async()
        if not model_name:
            console.print("[bold red]No custom model name entered. Exiting.[/bold red]")
            return ""
        return model_name
    
    return selected_model_or_custom

async def select_persona(console: Console, exclude: Optional[str] = None) -> str:
    """Select a persona."""
    options = list(PERSONAS.keys())
    
    if exclude and exclude in options:
        options.remove(exclude)
    
    console.print(Panel(
        "Select a computer scientist persona",
        title="[bold blue]Persona Selection[/bold blue]",
        border_style="blue"
    ))
    
    result = await questionary.select(
        "Choose a persona:",
        choices=options,
        use_arrow_keys=True,
        style=questionary.Style([
            ('selected', 'fg:#673ab7 bold'),
            ('highlighted', 'fg:#673ab7 bold')
        ])
    ).ask_async()
    
    return result

async def select_conversation_starter(console: Console) -> str:
    """Select or create a conversation starter."""
    console.print(Panel(
        "Select a conversation starter or create your own",
        title="[bold blue]Conversation Starter[/bold blue]",
        border_style="blue"
    ))
    
    starter = await questionary.select(
        "Choose a conversation starter:",
        choices=CONVERSATION_STARTERS,
        use_arrow_keys=True,
        style=questionary.Style([
            ('selected', 'fg:#673ab7 bold'),
            ('highlighted', 'fg:#673ab7 bold')
        ])
    ).ask_async()
    
    if starter == "Custom starter (type your own)...":
        custom_starter = await questionary.text(
            "Enter your custom conversation starter:"
        ).ask_async()
        return custom_starter if custom_starter and custom_starter.strip() else "What are your thoughts on the future of artificial intelligence?"
    
    return starter

async def select_max_turns(console: Console) -> int:
    """Select maximum conversation turns."""
    console.print(Panel(
        "Select the maximum number of conversation turns\n"
        "Use -1 for infinite conversation (press Ctrl+C to exit)",
        title="[bold blue]Conversation Length[/bold blue]",
        border_style="blue"
    ))
    
    def validate_turns(text):
        if not text.strip():
            return "Please enter a number"
        
        try:
            value = int(text)
            if value == -1 or value >= 2:
                return True
            return "Must be -1 (infinite) or at least 2 turns"
        except ValueError:
            return "Please enter a valid number"
    
    turns_text = await questionary.text(
        "Maximum conversation turns (-1 for infinite):",
        default="-1",
        validate=validate_turns
    ).ask_async()
    
    return int(turns_text)

async def main_conversation_loop(
    console: Console, 
    client: toc.TonicOllamaClient, 
    MODEL_NAME: str, 
    persona1_name: str, 
    persona2_name: str, 
    starter_prompt: str, 
    max_turns: int,
    progress_bar: Progress,
    overall_task
):
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="conversation_area"),
        Layout(name="progress_area", size=3),
        Layout(name="footer", size=1),
    )

    layout["header"].update(Panel(f"Conversation: {persona1_name} vs {persona2_name} | Model: {MODEL_NAME}", 
                                  title="[bold green]Live Chat[/bold green]", border_style="green"))
    layout["conversation_area"].split_row(
        Layout(name=persona1_name, ratio=1),
        Layout(name=persona2_name, ratio=1),
    )
    layout["progress_area"].update(progress_bar)
    layout["footer"].update(Text("Starting conversation...", justify="center"))
    
    # Generate conversation IDs
    persona1_conv_id = f"persona_{persona1_name.replace(' ', '_').lower()}_{get_timestamp()}"
    persona2_conv_id = f"persona_{persona2_name.replace(' ', '_').lower()}_{get_timestamp()}"
    
    conversation_log: List[Tuple[str, str]] = []

    # Initial prompt
    current_prompt = starter_prompt
    current_speaker = persona1_name
    other_speaker = persona2_name
    current_turn = 0

    persona1_generated_tokens = 0
    persona2_generated_tokens = 0
    
    # Setup display
    ai_message_text_content: List[Union[str, Text]] = [] 
    
    # Initialize panels
    layout[persona1_name].update(Panel(Text("Waiting..."), title=f"[bold blue]{persona1_name}[/bold blue]", border_style="blue", box=ROUNDED))
    layout[persona2_name].update(Panel(Text("Waiting..."), title=f"[bold magenta]{persona2_name}[/bold magenta]", border_style="magenta", box=ROUNDED))


    with Live(layout, console=console, refresh_per_second=10, vertical_overflow="visible") as live_display:
        try:
            while max_turns == -1 or current_turn < max_turns:
                # Swap speakers
                current_speaker, other_speaker = other_speaker, current_speaker
                current_speaker_conv_id = persona1_conv_id if current_speaker == persona1_name else persona2_conv_id
                
                current_turn += 1
                ai_message_text_content: List[Union[str, Text]] = []
                
                turn_info = f"Turn {current_turn}"
                if max_turns != -1:
                    turn_info += f"/{max_turns}"
                
                speaker_display_name = f"{current_speaker} (AI)"
                
                thinking_text = Text(f"{turn_info} - {speaker_display_name} is thinking...", style="bold yellow")
                current_speaker_panel_content = Group(thinking_text, Text.assemble(*ai_message_text_content))
                
                layout[current_speaker].update(Panel(
                    current_speaker_panel_content,
                    title=f"[bold blue]{current_speaker}[/bold blue]" if current_speaker == persona1_name else f"[bold magenta]{current_speaker}[/bold magenta]",
                    border_style="blue" if current_speaker == persona1_name else "magenta",
                    box=ROUNDED
                ))
                live_display.refresh()

                prompt_for_llm = current_prompt
                if current_speaker == persona1_name:
                    if persona1_generated_tokens > REPROMPT_TOKEN_THRESHOLD:
                        reminder = f"(System note to {persona1_name}: Remember your persona. You are {persona1_name}. The current topic, initiated by {other_speaker}, is: '{current_prompt[:100]}...')\n\n"
                        prompt_for_llm = reminder + current_prompt
                        persona1_generated_tokens = 0
                        if client.debug: toc.fancy_print(console, f"Reminding {persona1_name} of their persona.", style="dim cyan")
                elif current_speaker == persona2_name:
                    if persona2_generated_tokens > REPROMPT_TOKEN_THRESHOLD:
                        reminder = f"(System note to {persona2_name}: Remember your persona. You are {persona2_name}. The current topic, initiated by {other_speaker}, is: '{current_prompt[:100]}...')\n\n"
                        prompt_for_llm = reminder + current_prompt
                        persona2_generated_tokens = 0
                        if client.debug: toc.fancy_print(console, f"Reminding {persona2_name} of their persona.", style="dim cyan")
                
                full_ai_message_content = ""

                # Get streaming response
                response_stream = await client.chat(
                    model=MODEL_NAME,
                    message=prompt_for_llm,
                    temperature=0.8,
                    conversation_id=current_speaker_conv_id,
                    system_prompt=get_full_system_prompt(current_speaker),
                    stream=True
                )
                
                if isinstance(response_stream, AsyncGenerator):
                    async for partial_response in response_stream:
                        if hasattr(partial_response, 'message') and hasattr(partial_response.message, 'content') and partial_response.message.content:
                            chunk = partial_response.message.content
                            ai_message_text_content.append(chunk)
                            full_ai_message_content += chunk
                            
                            # Update display
                            current_speaker_panel_content = Group(thinking_text, Text.assemble(*ai_message_text_content))
                            layout[current_speaker].update(Panel(
                                current_speaker_panel_content,
                                title=f"[bold blue]{current_speaker}[/bold blue]" if current_speaker == persona1_name else f"[bold magenta]{current_speaker}[/bold magenta]",
                                border_style="blue" if current_speaker == persona1_name else "magenta",
                                box=ROUNDED
                            ))
                            live_display.refresh()
                        if hasattr(partial_response, 'done') and partial_response.done:
                            break
                else:
                    # Fallback for non-streaming (shouldn't happen)
                    if isinstance(response_stream, dict) and "message" in response_stream:
                        full_ai_message_content = response_stream["message"]["content"]
                        ai_message_text_content.append(full_ai_message_content)
                    else:
                        ai_message_text_content.append("Error: Unexpected response format.")
                    
                    current_speaker_panel_content = Group(Text(f"{turn_info} - {speaker_display_name}", style="bold green"), Text.assemble(*ai_message_text_content))
                    layout[current_speaker].update(Panel(
                        current_speaker_panel_content,
                        title=f"[bold blue]{current_speaker}[/bold blue]" if current_speaker == persona1_name else f"[bold magenta]{current_speaker}[/bold magenta]",
                        border_style="blue" if current_speaker == persona1_name else "magenta",
                        box=ROUNDED
                    ))
                    live_display.refresh()


                conversation_log.append((current_speaker, full_ai_message_content))
                
                # Track tokens
                generated_tokens_this_turn = count_tokens(full_ai_message_content)
                if current_speaker == persona1_name:
                    persona1_generated_tokens += generated_tokens_this_turn
                else:
                    persona2_generated_tokens += generated_tokens_this_turn

                # Update display with final content
                final_speaker_panel_content = Group(
                    Text(f"{turn_info} - {speaker_display_name}", style="bold green"),
                    Text.assemble(*ai_message_text_content)
                )
                layout[current_speaker].update(Panel(
                    final_speaker_panel_content,
                    title=f"[bold blue]{current_speaker}[/bold blue]" if current_speaker == persona1_name else f"[bold magenta]{current_speaker}[/bold magenta]",
                    border_style="blue" if current_speaker == persona1_name else "magenta",
                    box=ROUNDED
                ))
                live_display.refresh()
                
                current_prompt = full_ai_message_content
                
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            layout["footer"].update(Text("Conversation interrupted by user.", style="bold yellow", justify="center"))
        except Exception as e:
            layout["footer"].update(Text(f"An error occurred: {e}", style="bold red", justify="center"))
            if client.debug: toc.fancy_print(console, f"Error in conversation loop: {e}", style="red")
        finally:
            # Update progress bar
            if progress_bar and overall_task is not None:
                 progress_bar.update(overall_task, completed=max_turns if max_turns != -1 else current_turn, description="Conversation Ended", total=max_turns if max_turns != -1 else current_turn)
            
            layout["footer"].update(Text("Conversation ended. Log file saved.", style="bold green", justify="center"))
            live_display.refresh()
            
            log_filename = log_conversation_to_file(persona1_name, persona2_name, conversation_log, MODEL_NAME)
            console.print(f"\n[bold green]Conversation logged to: {log_filename}[/bold green]")


async def main():
    console = Console()
    
    # Welcome
    console.print(Panel(
        "Welcome to the Autonomous Conversation Demo\n"
        "This example will simulate a conversation between two historical computer scientists",
        title="[bold green]Tonic Ollama Client - Conversation Demo[/bold green]",
        border_style="green"
    ))
    
    # Select model
    MODEL_NAME = await select_model(console)
    if not MODEL_NAME:
        return
    
    # Check server
    client = toc.create_client(debug=False)
    try:
        await client.ensure_server_ready()
    except toc.OllamaServerNotRunningError:
        console.print("[bold red]Error: Ollama server is not running. Please start it with 'ollama serve'[/bold red]")
        return
    
    # Check model
    try:
        await client.get_async_client().show(model=MODEL_NAME)
    except toc.ResponseError as e:
        if e.status_code == 404:
            console.print(f"[bold yellow]Warning: Model '{MODEL_NAME}' not found.[/bold yellow]")
            console.print(f"[yellow]Please run: ollama pull {MODEL_NAME}[/yellow]")
            return
        else:
            console.print(f"[bold red]Error: {e}[/bold red]")
            return
    
    # Select personas and conversation parameters
    persona1 = await select_persona(console)
    if not persona1:
        return
    
    clear_console()
    
    persona2 = await select_persona(console, exclude=persona1)
    if not persona2:
        return
    
    clear_console()
    
    starter = await select_conversation_starter(console)
    if not starter:
        return
    
    max_turns = await select_max_turns(console)
    
    clear_console()

    # Setup progress bar
    progress_bar = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        expand=True,
    )
    
    overall_task = progress_bar.add_task(
        "Conversation Progress", 
        total=max_turns if max_turns != -1 else 100,
        start=(max_turns != -1)
    )
    if max_turns == -1:
        progress_bar.update(overall_task, description="Conversation (Ongoing - Ctrl+C to stop)")


    # Start conversation
    await main_conversation_loop(
        console, client, MODEL_NAME, 
        persona1, persona2, starter, max_turns,
        progress_bar, overall_task
    )

    # Cleanup
    await client.close(model_to_unload=MODEL_NAME)
    console.print(f"[bold blue]Client closed. Model {MODEL_NAME} unload attempted.[/bold blue]")

if __name__ == "__main__":
    asyncio.run(main())
