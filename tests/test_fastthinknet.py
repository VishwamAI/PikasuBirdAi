# Test script for FastThinkNet model
import unittest
import torch
from modules.fastthinknet_model import FastThinkNet

class TestFastThinkNet(unittest.TestCase):
    def test_initialization(self):
        """Test if the FastThinkNet model initializes without errors."""
        try:
            model = FastThinkNet()
            self.assertIsNotNone(model, "Failed to initialize FastThinkNet model.")
        except Exception as e:
            self.fail(f"Initialization of FastThinkNet model raised an exception: {e}")

    def test_forward_pass(self):
        """Test if the FastThinkNet model performs a forward pass without errors."""
        model = FastThinkNet()
        input_tensor = torch.randn(1, 1, 28, 28)  # Example input for MNIST
        try:
            output = model(input_tensor)
            self.assertEqual(output.shape, (1, 10), "Output shape mismatch.")
        except Exception as e:
            self.fail(f"Forward pass of FastThinkNet model raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()