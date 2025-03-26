from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from together import Together

class CustomLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        api_client,
        model_name
    ):
        self.model = Together()
        self.api_client = api_client
        self.model_name = model_name

    def load_model(self):
        return self.model

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt: str) -> str:

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        response = self.api_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            echo=True
        )

        output = response.choices[0].message.content

        return output

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)