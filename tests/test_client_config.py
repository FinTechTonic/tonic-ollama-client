import pytest
import pytest_asyncio
from unittest.mock import patch
from rich.console import Console # Ensure Console is imported directly
from tonic_ollama_client import TonicOllamaClient, create_client, ClientConfig

class TestClientConfiguration:
        
    def test_default_initialization(self):
        client = TonicOllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.max_server_startup_attempts == 3
        assert client.debug is False
        assert isinstance(client.console, Console)
        assert client.conversations == {}

    def test_custom_initialization(self):
        custom_console = Console()
        client = TonicOllamaClient(
            base_url="http://custom-url:12345",
            max_server_startup_attempts=5,
            debug=True,
            console=custom_console,
        )
        assert client.base_url == "http://custom-url:12345"
        assert client.max_server_startup_attempts == 5
        assert client.debug is True
        assert client.console == custom_console

    def test_client_config_model_defaults(self):
        config = ClientConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.max_server_startup_attempts == 3
        assert config.debug is False

    def test_client_config_model_custom(self):
        config = ClientConfig(
            base_url="http://another-url:54321",
            max_server_startup_attempts=10,
            debug=True
        )
        assert config.base_url == "http://another-url:54321"
        assert config.max_server_startup_attempts == 10
        assert config.debug is True

    def test_create_client_defaults(self):
        client = create_client()
        assert client.base_url == "http://localhost:11434"
        assert client.max_server_startup_attempts == 3
        assert client.debug is False
        assert isinstance(client.console, Console)

    def test_create_client_custom_params(self):
        custom_console = Console()
        client = create_client(
            base_url="http://custom-ollama:11434",
            max_server_startup_attempts=5,
            debug=True,
            console=custom_console
        )
        assert client.base_url == "http://custom-ollama:11434"
        assert client.max_server_startup_attempts == 5
        assert client.debug is True
        assert client.console == custom_console

    def test_client_config_passed_to_client(self):
        """Test TonicOllamaClient uses parameters as if from a config."""
        config = ClientConfig(
            base_url="http://config-test:1122", 
            max_server_startup_attempts=7,
            debug=True
        )
        
        client_from_config_values = TonicOllamaClient(
            base_url=config.base_url,
            max_server_startup_attempts=config.max_server_startup_attempts,
            debug=config.debug
        )

        assert client_from_config_values.base_url == config.base_url
        assert client_from_config_values.max_server_startup_attempts == config.max_server_startup_attempts
        assert client_from_config_values.debug == config.debug

    @patch('tonic_ollama_client.Console')
    def test_create_client_console_handling(self, MockConsole):
        """Test create_client console handling."""
        MockConsole.reset_mock()
        # When create_client is called without a console, it should create one.
        # The MockConsole passed to the test is patching the Console class within the tonic_ollama_client module.
        # So, when create_client internally calls Console(), it gets our MockConsole.
        client1 = create_client()
        MockConsole.assert_called_once() 
        # client1.console will be the instance returned by MockConsole()
        assert client1.console is MockConsole.return_value

        MockConsole.reset_mock()
        # When a console instance is passed, it should be used directly.
        my_console_instance = Console() # Create a real Console instance for this part
        client2 = create_client(console=my_console_instance)
        assert client2.console is my_console_instance
        MockConsole.assert_not_called() # Ensure the patched Console was not called to create a new one
