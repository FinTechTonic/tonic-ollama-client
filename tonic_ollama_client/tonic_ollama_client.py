"""
Tonic Ollama Client - A robust wrapper for Ollama API with enhanced functionality

This module provides a high-level client for interacting with Ollama models through
the Ollama API. It includes features like error handling, retries, conversation 
management, and support for various LLM operations.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union

# Third-party imports
from ollama import AsyncClient, ChatResponse, ResponseError
from rich.console import Console
from rich.panel import Panel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Message template for model not ready
OLLAMA_MODEL_NOT_READY_MESSAGE = """
[bold yellow]The Ollama model '{model_name}' is not ready.[/bold yellow]
Please ensure:
1. Ollama server is running (run 'ollama serve' in a separate terminal)
2. The model is pulled (run 'ollama pull {model_name}')

[bold cyan]Press Enter to retry, or 'q' to quit: [/bold cyan]"""


class TonicOllamaClient:
    """
    Enhanced Ollama client with built-in retries, validation, and interactive prompting
    for financial and regulatory document processing applications.
    
    Features:
    - Robust error handling with configurable retries
    - Conversation management
    - Interactive model readiness checks
    - Support for embedding generation
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        max_interactive_attempts: int = 3,
        debug: bool = False,
        console: Optional[Console] = None,
    ):
        """
        Initialize the TonicOllama client.

        Args:
            base_url: Ollama API base URL
            max_interactive_attempts: Maximum number of interactive attempts before giving up
            debug: Enable debug output
            console: Optional Rich Console instance for formatted output
        """
        self.base_url = base_url
        self.max_interactive_attempts = max_interactive_attempts
        self.debug = debug
        self.console = console or Console()
        self.async_client: Optional[AsyncClient] = None
        # Dictionary to store conversations by conversation_id
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        
        if self.debug:
            self.fancy_print(f"Initialized TonicOllamaClient with base_url={base_url}", style="dim blue")

    def fancy_print(
        self,
        message: str,
        style: Optional[str] = None,
        panel: bool = False,
        border_style: Optional[str] = None,
    ):
        """Print formatted messages using Rich console."""
        if panel:
            self.console.print(Panel(message, border_style=border_style or "blue"))
        else:
            self.console.print(message, style=style)

    def get_async_client(self) -> AsyncClient:
        """Get or create the async client instance"""
        if not self.async_client:
            self.async_client = AsyncClient(host=self.base_url)
        return self.async_client

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def check_model_ready(
        self, model_name: str, ready_prompt: str = "Respond with exactly one word: READY"
    ) -> bool:
        """
        Check if an Ollama model is ready with interactive user prompts.

        Args:
            model_name: Name of the Ollama model to check
            ready_prompt: Prompt to send to check model readiness

        Returns:
            bool: True if model is ready, False otherwise
        """
        consecutive_failures = 0
        client = self.get_async_client()

        while True:
            try:
                self.fancy_print(f"Checking if Ollama model '{model_name}' is ready...", style="yellow")

                response = await client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": ready_prompt}],
                    options={"temperature": 0.1},
                    stream=False,
                )

                if "READY" in response["message"]["content"].upper():
                    self.fancy_print(f"Ollama model '{model_name}' is responsive.", style="green")
                    return True
                else:
                    self.fancy_print(
                        f"Ollama model responded, but not with expected 'READY'. Response: {response['message']['content']}",
                        style="yellow",
                    )
                    consecutive_failures += 1
            except ResponseError as e:
                self.fancy_print(f"Ollama API error: {str(e)}", style="red")
                consecutive_failures += 1
            except Exception as e:
                self.fancy_print(f"Error checking model readiness: {str(e)}", style="red")
                consecutive_failures += 1

            if consecutive_failures >= self.max_interactive_attempts:
                self.fancy_print("Max retry attempts reached. Exiting.", style="red")
                return False

            # Prompt user for action
            self.fancy_print(
                OLLAMA_MODEL_NOT_READY_MESSAGE.format(model_name=model_name),
                style="yellow",
                panel=True,
            )

            user_input = await asyncio.to_thread(input().strip().lower)  # Make input non-blocking for async
            if user_input == "q":
                self.fancy_print("User requested to quit.", style="red")
                return False

            self.fancy_print("Retrying...", style="cyan")

    async def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Optional ID for the conversation. If not provided, a random ID will be generated.
            
        Returns:
            The conversation ID
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
            
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get the messages in a conversation.
        
        Args:
            conversation_id: ID of the conversation to retrieve
            
        Returns:
            List of messages in the conversation
            
        Raises:
            ValueError: If the conversation ID doesn't exist
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation with ID {conversation_id} does not exist")
            
        return self.conversations[conversation_id]
        
    def list_conversations(self) -> List[str]:
        """
        List all conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        return list(self.conversations.keys())
        
    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear a conversation's messages.
        
        Args:
            conversation_id: ID of the conversation to clear
            
        Raises:
            ValueError: If the conversation ID doesn't exist
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation with ID {conversation_id} does not exist")
            
        self.conversations[conversation_id] = []
        
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation.
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Raises:
            ValueError: If the conversation ID doesn't exist
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation with ID {conversation_id} does not exist")
            
        del self.conversations[conversation_id]

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def chat(
        self,
        model: str,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Union[Dict[str, Any], ChatResponse]:
        """
        Send a chat message to the model and get a response.
        
        Args:
            model: Name of the Ollama model to use
            message: User message to send
            conversation_id: ID of the conversation to use. If not provided, a new conversation will be created.
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Model response
        """
        # Create or retrieve conversation
        if conversation_id is None:
            conversation_id = await self.create_conversation()
        elif conversation_id not in self.conversations:
            await self.create_conversation(conversation_id)
        
        # Get conversation history
        messages = self.get_conversation(conversation_id).copy()
        
        # Add system prompt if provided and not already present
        if system_prompt and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Add user message
        user_message = {"role": "user", "content": message}
        messages.append(user_message)
        
        # Store user message in conversation history
        if not system_prompt or len(self.conversations[conversation_id]) > 0 or messages[0]["role"] != "system":
            self.conversations[conversation_id].append(user_message)
        
        try:
            client = self.get_async_client()
            
            if self.debug:
                self.fancy_print(f"Sending chat request to model '{model}'", style="dim blue")
            
            response = await client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
                stream=False,
            )
            
            # Store assistant response in conversation history
            assistant_message = {
                "role": "assistant", 
                "content": response["message"]["content"]
            }
            self.conversations[conversation_id].append(assistant_message)
            
            if self.debug:
                self.fancy_print(f"Received response from model '{model}'", style="dim blue")
            
            return response
                
        except ResponseError as e:
            self.fancy_print(f"Ollama API error: {str(e)}", style="red")
            raise
        except Exception as e:
            self.fancy_print(f"Error in chat: {str(e)}", style="red")
            raise

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def generate_embedding(self, model: str, text: str) -> List[float]:
        """
        Generate embeddings for the given text.
        
        Args:
            model: Name of the Ollama model to use
            text: Text to generate embeddings for
            
        Returns:
            List of embedding values
        """
        try:
            client = self.get_async_client()
            
            if self.debug:
                self.fancy_print(f"Generating embeddings with model '{model}'", style="dim blue")
                
            response = await client.embeddings(model=model, prompt=text)
            
            if self.debug:
                self.fancy_print(f"Generated embeddings with {len(response['embedding'])} dimensions", style="dim blue")
                
            return response["embedding"]
        except ResponseError as e:
            self.fancy_print(f"Ollama API error: {str(e)}", style="red")
            raise
        except Exception as e:
            self.fancy_print(f"Error generating embeddings: {str(e)}", style="red")
            raise


# Utility function to create a pre-configured client
def create_client(
    base_url: str = "http://localhost:11434",
    debug: bool = False,
    console: Optional[Console] = None,
) -> TonicOllamaClient:
    """
    Create a pre-configured TonicOllama client instance
    
    Args:
        base_url: Ollama API base URL
        debug: Enable debug output
        console: Optional Rich Console instance for formatted output
        
    Returns:
        Configured TonicOllamaClient instance
    """
    return TonicOllamaClient(
        base_url=base_url,
        debug=debug,
        console=console,
    )
