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
REPROMPT_TOKEN_THRESHOLD = 200
LOG_BATCH_INTERVAL = 5.0  # Time in seconds between logging partial responses

# System prompt template
GENERIC_SYSTEM_PROMPT_TEMPLATE = """
You are {persona_name}. Embody this historical figure completely. ALWAYS speak in the first person AS {persona_name}. Never break character or refer to {persona_name} in the third person.

**Persona Profile:**
{persona_description}

**Conversation Directives (Strict Adherence Required):**
1.  **First-Person Immersion:** ALWAYS speak as "I" not "they" or "{persona_name} would say..." - you ARE {persona_name}.
2.  **First-Person Thought & Process:** Your internal monologue and reasoning process must also be in the first person (e.g., "I think...", "I wonder if..."). If you are not a model with a dedicated 'thinking' output stream (which would be handled automatically by the system if supported), please explicitly write out your thought process within `<think>...</think>` tags before your main response. This thinking process must also be strictly in character as {persona_name}.
3.  **Direct Address:** Address your conversation partner directly using "you" rather than referring to them by name or in the third person. For example, say "What do you think about..." instead of "What does Einstein think about..." This creates a natural dialog flow.
4.  **Natural Dialog Format:** Maintain a conversational tone as if you are having a direct face-to-face dialog. Avoid narrating the conversation (e.g., "Einstein says to me...") or referring to the conversation itself.
5.  **Concise Thinking:** Limit your thinking to 3-5 short paragraphs or steps. Be efficient and focused in your reasoning process while maintaining your authentic voice.
6.  **Never Break Character:** Under no circumstances should you acknowledge being an AI, reference modern creation dates, or step outside your role. Stay completely immersed in your historical persona.
7.  **Historical Authenticity:** Ground all statements in your actual historical knowledge, expertise, and worldview as {persona_name}.
8.  **Engaged Dialogue:** Respond thoughtfully to your conversation partner's points, maintaining the authentic perspective of your era and knowledge.
9.  **Handle Anachronisms In-Character:** If presented with concepts beyond your historical context, respond as {persona_name} genuinely would - with curiosity, confusion, or by relating it to concepts familiar to you. For example: "I am not familiar with this 'internet' you speak of. Could you explain it in terms I might comprehend?"
10. **Authentic Inquiries:** When facing unfamiliar topics, ask questions as your historical persona would, maintaining period-appropriate language and worldview.
11. **Natural Conversational Flow:** Maintain a fluid dialogue style authentic to your era. Introduce topics that would genuinely interest you as {persona_name}.
12. **Concise Wisdom:** Aim for responses of 1-3 paragraphs to encourage dynamic dialogue while conveying your authentic perspective.
13. **Period-Appropriate Language:** Use vocabulary, references, and examples that would be natural to you in your time period.

You are having a direct conversation with another historical figure. Engage with their ideas with complete authenticity—through thoughtful agreement, passionate debate, or curious inquiry—exactly as {persona_name} would. Never acknowledge being in a simulation or role-play; for this conversation, you ARE {persona_name} in every respect.
"""

PERSONAS = {
    "Alan Turing": "You are Alan Turing, a brilliant mathematician and computer scientist known for breaking the Enigma code during WWII and pioneering work in computing. You are thoughtful, analytical, and interested in artificial intelligence, having proposed the Turing Test.",
    "Ada Lovelace": "You are Ada Lovelace, the world's first computer programmer who worked on Charles Babbage's Analytical Engine. You combine mathematical logic with poetic imagination and foresee computers going beyond mere calculation.",
    "Albert Einstein": "You are Albert Einstein, the revolutionary physicist who developed the theory of relativity. You approach problems with thought experiments and visual imagination. You're philosophical, pacifistic, and believe in simplicity. You often use analogies and thought experiments to explain complex ideas.",
    "Charles Darwin": "You are Charles Darwin, the naturalist who proposed the theory of evolution by natural selection. You are methodical, observant, and cautious. You collect extensive evidence before drawing conclusions and are willing to challenge established beliefs when evidence demands it.",
    "Marie Curie": "You are Marie Curie, a pioneering physicist and chemist who discovered radium and polonium and conducted groundbreaking research on radioactivity. You are determined, meticulous, and dedicated to scientific inquiry despite significant obstacles. You believe in the practical application of scientific discoveries for the betterment of humanity.",
    "Nikola Tesla": "You are Nikola Tesla, an eccentric inventor and electrical engineer who developed the alternating current electrical system. You have a photographic memory and powerful visualization abilities. Your mind works in flashes of insight and you're often decades ahead of your contemporaries in your thinking, with a particular fascination with wireless energy transmission.",
    "Richard Feynman": "You are Richard Feynman, a Nobel Prize-winning theoretical physicist known for your work in quantum mechanics and particle physics. You combine brilliant scientific insight with playful curiosity and a gift for explaining complex concepts simply. You believe in the joy of discovery and the importance of not fooling yourself in scientific inquiry.",
    "Aristotle": "You are Aristotle, the ancient Greek philosopher who made foundational contributions to logic, metaphysics, ethics, and natural sciences. You approach knowledge systematically, categorizing, and analyzing concepts. You believe in empirical observation and logical reasoning as paths to understanding.",
    "Leonardo da Vinci": "You are Leonardo da Vinci, the Renaissance polymath whose interests spanned anatomy, engineering, astronomy, botany, and more. Your approach combines scientific precision with artistic sensibility. You are endlessly curious, filling notebooks with observations, questions, and detailed sketches to understand the world's patterns and mechanisms."
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
DEFAULT_AVAILABLE_MODELS = ["qwen3:8b", "llama3.1:latest", "phi4:latest", "mistral:latest"]
CUSTOM_MODEL_OPTION = "Enter Custom Model Name..."
PROGRESS_STEPS = [f"Step {i}" for i in range(20)]


def clear_console():
    """Clear the console screen."""
    # This function is now a no-op to prevent flickering
    # Using Rich's Live display will handle updates without needing to clear
    pass

def get_timestamp():
    """Get current timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def create_log_file_header(persona1: str, persona2: str, model_name: str) -> str:
    """Create the header for the log file and return the filename."""
    epoch_timestamp = int(datetime.datetime.now().timestamp())
    filename = f"convo_{epoch_timestamp}.log"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Conversation between {persona1} and {persona2}\n")
        f.write(f"# Model: {model_name}\n")
        f.write(f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#" + "="*79 + "\n\n")
    
    # Create a separate debug log if needed for partial responses
    if os.environ.get("SAVE_PARTIAL_RESPONSES", "0") == "1":
        debug_filename = f"convo_{epoch_timestamp}_debug.log"
        with open(debug_filename, 'w', encoding='utf-8') as f:
            f.write(f"# DEBUG LOG - Partial Responses\n")
            f.write(f"# Conversation between {persona1} and {persona2}\n")
            f.write(f"# Model: {model_name}\n")
            f.write(f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("#" + "="*79 + "\n\n")
    else:
        debug_filename = None
    
    return filename

def append_to_log_file(filename: str, speaker: str, message: str, thinking: str = "", partial: bool = False):
    """
    Append a single message to the log file.
    
    Args:
        filename: Path to the log file
        speaker: Name of the speaker
        message: Content of the message
        thinking: Thinking content if available
        partial: If True, marks this as a partial response (streaming)
    """
    # For partial responses, only log to debug file if enabled
    if partial:
        debug_filename = filename.replace(".log", "_debug.log")
        if os.environ.get("SAVE_PARTIAL_RESPONSES", "0") == "1" and os.path.exists(debug_filename):
            with open(debug_filename, 'a', encoding='utf-8') as f:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(f"[PARTIAL {timestamp}] <{speaker}>\n{message}\n</>\n\n")
                f.flush()
                os.fsync(f.fileno())
        return
    
    # For complete responses, log in ChatML-like format with thinking content
    with open(filename, 'a', encoding='utf-8') as f:
        if thinking:
            f.write(f"<{speaker}>\n<think>\n{thinking}\n</think>\n{message}\n</>\n\n")
        else:
            f.write(f"<{speaker}>\n{message}\n</>\n\n")
        f.flush()
        os.fsync(f.fileno())

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
    # Use epoch timestamp for the filename
    epoch_timestamp = int(datetime.datetime.now().timestamp())
    filename = f"convo_{epoch_timestamp}.log"
    
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

    # Find default model index
    default_index = next((i for i, m in enumerate(model_choices) if "qwen3" in m.lower()), 0)

    selected_model_or_custom = await questionary.select(
        "Please select a model to use for this conversation:",
        choices=model_choices,
        default=model_choices[default_index],
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
    
    # Add "Random" option at the beginning
    options.insert(0, "Random (Choose for me)")
    
    if exclude and exclude in options:
        options.remove(exclude)
    
    console.print(Panel(
        "Select a historical persona",
        title="[bold blue]Persona Selection[/bold blue]",
        border_style="blue"
    ))
    
    result = await questionary.select(
        "Choose a persona:",
        choices=options,
        default="Random (Choose for me)",
        use_arrow_keys=True,
        style=questionary.Style([
            ('selected', 'fg:#673ab7 bold'),
            ('highlighted', 'fg:#673ab7 bold')
        ])
    ).ask_async()
    
    # Handle random selection
    if result == "Random (Choose for me)":
        import random
        persona_options = list(PERSONAS.keys())
        if exclude and exclude in persona_options:
            persona_options.remove(exclude)
        result = random.choice(persona_options)
        console.print(f"[bold green]Randomly selected: {result}[/bold green]")
    
    return result

async def select_conversation_starter(console: Console) -> str:
    """Select or create a conversation starter."""
    console.print(Panel(
        "Select a conversation starter or create your own",
        title="[bold blue]Conversation Starter[/bold blue]",
        border_style="blue"
    ))
    
    # Create options list with Random as first option
    options = ["Random (Choose for me)"]
    options.extend(CONVERSATION_STARTERS)
    
    starter = await questionary.select(
        "Choose a conversation starter:",
        choices=options,
        default="Random (Choose for me)",
        use_arrow_keys=True,
        style=questionary.Style([
            ('selected', 'fg:#673ab7 bold'),
            ('highlighted', 'fg:#673ab7 bold')
        ])
    ).ask_async()
    
    # Handle random selection
    if starter == "Random (Choose for me)":
        import random
        # Get all options except "Random" and "Custom"
        available_starters = [s for s in CONVERSATION_STARTERS if s != "Custom starter (type your own)..."]
        starter = random.choice(available_starters)
        console.print(f"[bold green]Randomly selected: [/bold green][italic]\"{starter}\"[/italic]")
    
    # Handle custom starter
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
    
    # Fix persona positions - persona1 on left, persona2 on right
    layout["conversation_area"].split_row(
        Layout(name=persona1_name, ratio=1),
        Layout(name=persona2_name, ratio=1),
    )
    
    layout["progress_area"].update(progress_bar)
    layout["footer"].update(Text(f"Tokens/sec: 0.0 | Total tokens: 0", justify="center"))
    
    # Generate conversation IDs
    persona1_conv_id = f"persona_{persona1_name.replace(' ', '_').lower()}_{get_timestamp()}"
    persona2_conv_id = f"persona_{persona2_name.replace(' ', '_').lower()}_{get_timestamp()}"
    
    conversation_log: List[Tuple[str, str]] = [] # For logging to file
    
    # Create log file with header
    log_filename = create_log_file_header(persona1_name, persona2_name, MODEL_NAME)
    
    # Reset the log file - ensure we start fresh
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(f"# Conversation between {persona1_name} and {persona2_name}\n")
        f.write(f"# Model: {MODEL_NAME}\n")
        f.write(f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#" + "="*79 + "\n\n")
    
    # Initialize a flag to track if the current turn's response has been logged
    current_turn_logged = False
    
    # Initial prompt
    current_prompt = starter_prompt
    
    # Initialize speakers - persona1 is ALWAYS first speaker (left panel)
    current_speaker = persona2_name  # Set to persona2 initially
    other_speaker = persona1_name    # Set to persona1 initially
    
    current_turn = 0

    persona1_generated_tokens = 0
    persona2_generated_tokens = 0
    
    # Initialize token rate tracking variables
    total_tokens_generated = 0
    token_rate_start_time = datetime.datetime.now()
    current_token_rate = 0.0
    token_rate_update_interval = 2.0  # seconds between token rate updates
    last_token_rate_update = datetime.datetime.now()
    
    # Track accumulated content for streaming log batches - moved earlier to ensure initialization
    accumulated_response = ""
    # Initialize last_log_time at function start to avoid the error
    last_log_time = datetime.datetime.now()
    
    # Setup Rich Live display
    # ai_message_text_content will store Text objects for the current turn's display
    live_display_content_elements: List[Text] = []
    
    # Adjust truncation based on console width
    def calculate_max_display_length():
        """Dynamically calculate max display length based on console dimensions"""
        # Reduce base value to make truncation happen sooner
        base_length = 800
        
        # Adjust based on console width - wider consoles can display more text
        # Get the actual console width, with a reasonable default if detection fails
        try:
            width = console.width or 80
            # Reduce the multiplier to make truncation happen sooner
            return min(max(base_length, width * 10), 2000)
        except Exception:
            return base_length
    
    # Set initial value but will recalculate during display updates
    MAX_DISPLAY_LENGTH = calculate_max_display_length()
    
    # Initialize layout and panels for personas
    layout[persona1_name].update(Panel(Text("Waiting..."), title=f"[bold blue]{persona1_name}[/bold blue]", border_style="blue", box=ROUNDED))
    layout[persona2_name].update(Panel(Text("Waiting..."), title=f"[bold magenta]{persona2_name}[/bold magenta]", border_style="magenta", box=ROUNDED))

    # Add throttling control for display updates
    last_refresh_time = datetime.datetime.now()
    MIN_REFRESH_INTERVAL = 0.1  # Seconds between display refreshes
    
    # Track content changes to avoid unnecessary refreshes
    previous_thinking_content = ""
    previous_message_content = ""
    
    with Live(layout, console=console, refresh_per_second=2, vertical_overflow="visible", auto_refresh=False) as live_display:
        try:
            while max_turns == -1 or current_turn < max_turns:
                # Swap speakers - this ensures persona1 (left panel) goes first
                # since we initialized current_speaker to persona2 above
                current_speaker, other_speaker = other_speaker, current_speaker
                current_speaker_conv_id = persona1_conv_id if current_speaker == persona1_name else persona2_conv_id
                
                # Reset logging flag for this turn
                current_turn_logged = False
                
                current_turn += 1
                live_display_content_elements.clear() # Clear for the new turn
                
                turn_info = f"Turn {current_turn}"
                if max_turns != -1:
                    turn_info += f"/{max_turns}"
                
                speaker_display_name = f"{current_speaker} (AI)"
                
                thinking_label_text = Text(f"{turn_info} - {speaker_display_name} is thinking...", style="bold yellow")
                # Initial panel content before streaming starts
                current_speaker_panel_content = Group(thinking_label_text)
                
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
                
                # Reset accumulators for this turn
                full_ai_message_content = ""
                full_ai_thinking_content = ""
                
                # Determine if we expect 'thinking' field from qwen3 models
                expect_separate_thinking_field = "qwen3" in MODEL_NAME.lower()

                # Always use streaming
                response_stream = await client.chat(
                    model=MODEL_NAME,
                    message=prompt_for_llm, # Use potentially modified prompt
                    temperature=0.8,
                    conversation_id=current_speaker_conv_id, # Use persona-specific conversation ID
                    system_prompt=get_full_system_prompt(current_speaker), # Ensure system prompt is for current speaker
                    stream=True
                    # think parameter is now handled by the client internally
                )
                
                if not isinstance(response_stream, AsyncGenerator):
                    # This case should ideally not be reached if client.chat(stream=True) behaves as expected.
                    # If it's not an AsyncGenerator, it's an unexpected error or a non-streaming dict response.
                    error_message = "Error: Expected a streaming response (AsyncGenerator) but received a different type."
                    if isinstance(response_stream, dict): # It might be a non-streaming dict response
                        error_message = f"Error: Expected a streaming response, but received a non-streaming dict: {str(response_stream)[:200]}"
                    
                    toc.fancy_print(console, error_message, style="bold red")
                    layout["footer"].update(Text(error_message, style="bold red", justify="center"))
                    # Potentially break or raise an exception here depending on desired error handling
                    break # Exit the conversation loop

                # Fix to prevent early termination: Ensure we always have a full response
                generated_content = False
                
                async for partial_response in response_stream:
                    # Only process display updates if content has changed
                    content_changed = False
                    
                    # Process thinking if present
                    if expect_separate_thinking_field and hasattr(partial_response, 'message') and hasattr(partial_response.message, 'thinking') and partial_response.message.thinking:
                        # Append new thinking content
                        new_thinking = partial_response.message.thinking
                        if new_thinking != full_ai_thinking_content[-len(new_thinking):] if full_ai_thinking_content else True:
                            full_ai_thinking_content += new_thinking
                            content_changed = True
                    
                    # Process main content
                    if hasattr(partial_response, 'message') and hasattr(partial_response.message, 'content') and partial_response.message.content:
                        chunk = partial_response.message.content
                        if chunk:  # Only update if there's actual content
                            full_ai_message_content += chunk
                            content_changed = True
                            generated_content = True  # Mark that we've gotten actual content
                        
                        # Update accumulated content for logging
                        accumulated_response = full_ai_message_content
                        
                        # Only log partial response every LOG_BATCH_INTERVAL seconds and only if debug logging is enabled
                        current_time = datetime.datetime.now()
                        time_since_last_log = (current_time - last_log_time).total_seconds()
                        
                        if time_since_last_log >= LOG_BATCH_INTERVAL:
                            append_to_log_file(log_filename, current_speaker, accumulated_response, partial=True)
                            last_log_time = current_time
                            
                            if client.debug:
                                toc.fancy_print(console, f"Logged partial response to debug log ({len(accumulated_response)} chars)", style="dim cyan")
                    
                    # Only update display if content changed and enough time has passed
                    current_time = datetime.datetime.now()
                    time_since_refresh = (current_time - last_refresh_time).total_seconds()
                    
                    if (content_changed and time_since_refresh >= MIN_REFRESH_INTERVAL) or hasattr(partial_response, 'done') and partial_response.done:
                        # Recalculate truncation length based on current console dimensions
                        MAX_DISPLAY_LENGTH = calculate_max_display_length()
                        
                        # Update live display content elements only when needed
                        live_display_content_elements.clear()
                        
                        # First display thinking content if available
                        if full_ai_thinking_content and full_ai_thinking_content != previous_thinking_content:
                            thinking_display = full_ai_thinking_content
                            if len(thinking_display) > MAX_DISPLAY_LENGTH:
                                thinking_display = "..." + thinking_display[-(MAX_DISPLAY_LENGTH-3):]
                            # Remove the thinking label, but keep styling
                            live_display_content_elements.append(Text(thinking_display, style="italic dim yellow"))
                            live_display_content_elements.append(Text("\n"))  # Add separator between thinking and response
                            previous_thinking_content = full_ai_thinking_content
                        
                        # Then display the actual response content with different styling
                        if full_ai_message_content and full_ai_message_content != previous_message_content:
                            # Remove the response label, just show content
                            message_display = full_ai_message_content
                            if len(message_display) > MAX_DISPLAY_LENGTH:
                                message_display = "..." + message_display[-(MAX_DISPLAY_LENGTH-3):]
                            live_display_content_elements.append(Text(message_display, style="bold white"))
                            previous_message_content = full_ai_message_content
                        
                        # Update label based on whether we're still thinking or done
                        if hasattr(partial_response, 'done') and partial_response.done:
                            turn_label = Text(f"{turn_info} - {speaker_display_name}", style="bold green")
                        else:
                            turn_label = thinking_label_text
                            
                        current_speaker_panel_content = Group(turn_label, *live_display_content_elements)
                        layout[current_speaker].update(Panel(
                            current_speaker_panel_content,
                            title=f"[bold blue]{current_speaker}[/bold blue]" if current_speaker == persona1_name else f"[bold magenta]{current_speaker}[/bold magenta]",
                            border_style="blue" if current_speaker == persona1_name else "magenta",
                            box=ROUNDED
                        ))
                        
                        # Force refresh and update timestamp
                        live_display.refresh()
                        last_refresh_time = current_time
                    
                    # Check if complete
                    if hasattr(partial_response, 'done') and partial_response.done:
                        # Final log of the complete response with thinking content
                        if not current_turn_logged:
                            append_to_log_file(
                                log_filename, 
                                current_speaker, 
                                full_ai_message_content,
                                thinking=full_ai_thinking_content, 
                                partial=False
                            )
                            current_turn_logged = True
                        break
                
                # If thinking finished but no actual content was generated, continue waiting
                if not generated_content and full_ai_thinking_content and not full_ai_message_content:
                    if client.debug:
                        toc.fancy_print(console, f"Warning: Thinking completed but no content generated. Continuing to wait...", style="yellow")
                    
                    # Wait for a small delay and continue the loop with the same speaker
                    await asyncio.sleep(0.5)
                    # Don't swap speakers for the next iteration
                    current_speaker, other_speaker = other_speaker, current_speaker
                    continue
                
                # Move the remaining code outside the streaming loop
                conversation_log.append((current_speaker, full_ai_message_content))
                
                # Ensure the final response is logged - if not logged during streaming
                if not current_turn_logged:
                    append_to_log_file(
                        log_filename, 
                        current_speaker, 
                        full_ai_message_content,
                        thinking=full_ai_thinking_content, 
                        partial=False
                    )
                    current_turn_logged = True
                
                # Calculate final tokens for this turn
                generated_tokens_this_turn = count_tokens(full_ai_message_content)
                if full_ai_thinking_content: # Add thinking tokens if they came separately
                    generated_tokens_this_turn += count_tokens(full_ai_thinking_content)

                # Update total tokens count
                total_tokens_generated += generated_tokens_this_turn
                
                # Recalculate token rate
                current_time = datetime.datetime.now()
                time_elapsed = (current_time - token_rate_start_time).total_seconds()
                if time_elapsed > 0:
                    current_token_rate = total_tokens_generated / time_elapsed
                
                # Update the footer with final token rate for this turn
                layout["footer"].update(Text(
                    f"Tokens/sec: {current_token_rate:.1f} | Total tokens: {total_tokens_generated:,}",
                    justify="center", style="bold cyan"
                ))
                
                if current_speaker == persona1_name:
                    persona1_generated_tokens += generated_tokens_this_turn
                else:
                    persona2_generated_tokens += generated_tokens_this_turn

                # Final display for the turn
                # Recalculate truncation based on current console width
                MAX_DISPLAY_LENGTH = calculate_max_display_length()
                
                final_display_elements_for_panel = [Text(f"{turn_info} - {speaker_display_name}", style="bold green")]
                
                # Add thinking section if available (without label)
                if full_ai_thinking_content:
                    thinking_display = full_ai_thinking_content
                    if len(thinking_display) > MAX_DISPLAY_LENGTH:
                        thinking_display = "..." + thinking_display[-(MAX_DISPLAY_LENGTH-3):]
                    final_display_elements_for_panel.append(Text(thinking_display, style="italic dim yellow"))
                    final_display_elements_for_panel.append(Text("\n"))  # Add separator
                
                # Add response section (without label)
                if full_ai_message_content:
                    message_display = full_ai_message_content
                    if len(message_display) > MAX_DISPLAY_LENGTH:
                        message_display = "..." + message_display[-(MAX_DISPLAY_LENGTH-3):]
                    final_display_elements_for_panel.append(Text(message_display, style="bold white"))
                
                final_speaker_panel_content = Group(*final_display_elements_for_panel)
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
            # Ensure partial response is logged if interrupted
            if accumulated_response and not current_turn_logged:
                append_to_log_file(
                    log_filename, 
                    current_speaker, 
                    accumulated_response,
                    thinking=full_ai_thinking_content, 
                    partial=False
                )
                append_to_log_file(log_filename, "SYSTEM", "Conversation interrupted by user.", partial=False)
            
            layout["footer"].update(Text("Conversation interrupted by user.", style="bold yellow", justify="center"))
        except Exception as e:
            # Ensure partial response is logged if error occurs
            if accumulated_response and not current_turn_logged:
                append_to_log_file(
                    log_filename, 
                    current_speaker, 
                    accumulated_response,
                    thinking=full_ai_thinking_content, 
                    partial=False
                )
                append_to_log_file(log_filename, "SYSTEM", f"Error occurred: {str(e)}", partial=False)
            
            layout["footer"].update(Text(f"An error occurred: {e}", style="bold red", justify="center"))
            if client.debug: 
                toc.fancy_print(console, f"Error in conversation loop: {e}", style="red")
        finally:
            # Update progress bar
            if progress_bar and overall_task is not None:
                 progress_bar.update(overall_task, completed=max_turns if max_turns != -1 else current_turn, description="Conversation Ended", total=max_turns if max_turns != -1 else current_turn)
            
            # Update footer with final token statistics
            elapsed_time = (datetime.datetime.now() - token_rate_start_time).total_seconds()
            final_token_rate = total_tokens_generated / elapsed_time if elapsed_time > 0 else 0
            layout["footer"].update(Text(
                f"Conversation ended. Avg tokens/sec: {final_token_rate:.1f} | Total tokens: {total_tokens_generated:,} | Log: {log_filename}",
                justify="center", style="bold green"
            ))
            live_display.refresh()
            
            # No need to create a new log file since we've been updating it incrementally
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
    
    # Don't clear console between selections to prevent flickering
    # clear_console()  # Removed
    
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
    
    # clear_console()  # Removed
    
    persona2 = await select_persona(console, exclude=persona1)
    if not persona2:
        return
    
    # clear_console()  # Removed
    
    starter = await select_conversation_starter(console)
    if not starter:
        return
    
    max_turns = await select_max_turns(console)
    
    # Use a transition panel instead of clearing the console
    console.print(Panel(
        f"Starting conversation between [bold blue]{persona1}[/bold blue] and [bold magenta]{persona2}[/bold magenta]...\n"
        f"Using model: [bold cyan]{MODEL_NAME}[/bold cyan]",
        title="[bold green]Conversation Setup Complete[/bold green]",
        border_style="green"
    ))
    
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
