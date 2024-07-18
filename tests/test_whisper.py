# Test script for OpenAI Whisper model
import unittest
from modules.whisper_model import WhisperModel, ModelDimensions

class TestWhisperModel(unittest.TestCase):
    def test_initialization(self):
        """Test if the WhisperModel initializes without errors."""
        try:
            dims = ModelDimensions(
                n_mels=80,
                n_audio_ctx=1500,
                n_audio_state=384,
                n_audio_head=6,
                n_audio_layer=4,
                n_vocab=51865,
                n_text_ctx=448,
                n_text_state=384,
                n_text_head=6,
                n_text_layer=4
            )
            model = WhisperModel(dims)
            self.assertIsNotNone(model, "Failed to initialize WhisperModel.")
        except Exception as e:
            self.fail(f"Initialization of WhisperModel raised an exception: {e}")

    # Add more tests as needed for the WhisperModel

if __name__ == '__main__':
    unittest.main()