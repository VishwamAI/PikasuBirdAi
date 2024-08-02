import os
import requests
from routellm import RouteLLMController

class ResponseGenerator:
    def __init__(self):
        # Initialize RouteLLM controller
        self.controller = RouteLLMController(
            router="gpt-3.5-turbo",
            models=["gpt-3.5-turbo", "gpt-4"],
            api_key=os.environ.get('ROUTELLM_API_KEY')
        )
        # Initialize ollama API endpoint
        self.ollama_api_url = "http://localhost:11434/api/generate"

    def generate_response(self, threat_level):
        # Generate a response based on the threat level using RouteLLM
        prompt = self._create_prompt(threat_level)
        response = self.controller.chat.completions.create(
            model="router:0.01",  # Use router with 0.01 cost threshold
            messages=[{"role": "system", "content": "You are an AI assistant for threat assessment and response generation."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def generate_response_ollama(self, threat_level):
        # Generate a response based on the threat level using ollama
        prompt = self._create_prompt(threat_level)
        payload = {
            "model": "llama3.1",
            "prompt": prompt
        }
        response = requests.post(self.ollama_api_url, json=payload)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: Unable to generate response. Status code: {response.status_code}"

    def _create_prompt(self, threat_level):
        if threat_level < 0.3:
            return "Generate a response for a low threat level situation."
        elif threat_level < 0.7:
            return "Generate a response for a moderate threat level situation."
        else:
            return "Generate a response for a high threat level situation."
