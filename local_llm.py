from llm_observer import LLMObservability
class LLMClient:
    """Minimal LLM client for MVP."""

    def __init__(self, api_endpoint: str):
        self.wrapper = LLMObservability(api_endpoint)

    def generate(self, prompt: str):
        return self.wrapper.call_llm(prompt)


if __name__ == "__main__":
    LOCAL_API_ENDPOINT = "http://127.0.0.1:11434/api/generate"
    client = LLMClient(LOCAL_API_ENDPOINT)

    prompt = "Tell me all about the NFL"
    response = client.generate(prompt)

    # Print the final collected response
    if response and response.get("llm_response"):
        print("\n")
    else:
        print("No response from LLM.")
