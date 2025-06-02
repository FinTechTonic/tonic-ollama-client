from unittest.mock import patch
from rich.console import Console
from tonic_ollama_client import TonicOllamaClient, create_client, ClientConfig

class TestClientConfiguration:
        
    def test_default_initialization(self):
        client = TonicOllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.max_server_startup_attempts == 3
        assert client.debug is False
        assert client.concurrent_models == 1  # Should always be 1
        assert isinstance(client.console, Console)
        assert client.conversations == {}
        assert client.stream_responses is False # Check default for stream_responses

    def test_custom_initialization(self):
        custom_console = Console()
        client = TonicOllamaClient(
            base_url="http://custom-url:12345",
            max_server_startup_attempts=5,
            debug=True,
            console=custom_console,
            concurrent_models=2,  # Should be ignored and set to 1
            stream_responses=True,
        )
        assert client.base_url == "http://custom-url:12345"
        assert client.max_server_startup_attempts == 5
        assert client.debug is True
        assert client.concurrent_models == 1  # Should always be 1, ignoring the passed value
        assert client.console == custom_console
        assert client.stream_responses is True

    def test_create_client_defaults(self):
        client = create_client()
        assert client.base_url == "http://localhost:11434"
        assert client.max_server_startup_attempts == 3
        assert client.debug is False
        assert isinstance(client.console, Console)
        assert client.stream_responses is False

    def test_create_client_custom_params(self):
        custom_console = Console()
        client = create_client(
            base_url="http://custom-ollama:11434",
            max_server_startup_attempts=5,
            debug=True,
            console=custom_console,
            concurrent_models=4,  # Should be ignored and set to 1
            stream_responses=True,
        )
        assert client.base_url == "http://custom-ollama:11434"
        assert client.max_server_startup_attempts == 5
        assert client.debug is True
        assert client.concurrent_models == 1  # Should always be 1, ignoring the passed value
        assert client.console == custom_console
        assert client.stream_responses is True

    def test_client_config_passed_to_client(self):
        """Test TonicOllamaClient uses parameters as if from a config."""
        config = ClientConfig(
            base_url="http://config-test:1122", 
            max_server_startup_attempts=7,
            debug=True,
            stream_responses=True,
        )
        
        client_from_config_values = TonicOllamaClient(
            base_url=config.base_url,
            max_server_startup_attempts=config.max_server_startup_attempts,
            debug=config.debug,
            stream_responses=config.stream_responses,
        )

        assert client_from_config_values.base_url == config.base_url
        assert client_from_config_values.max_server_startup_attempts == config.max_server_startup_attempts
        assert client_from_config_values.debug == config.debug
        assert client_from_config_values.stream_responses == config.stream_responses

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
