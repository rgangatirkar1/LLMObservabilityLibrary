import requests
import time
import json

class LLMObservability:
    """
    Minimal LLM wrapper for streaming output + background metric collection.
    """

    def __init__(self, api_endpoint: str, model: str = "mistral"):
        if not api_endpoint:
            raise ValueError("API endpoint cannot be empty.")
        self.api_endpoint = api_endpoint
        self.model = model
        self.session = requests.Session()

    def call_llm(self, prompt: str, stream: bool = True):
        """
        Stream the LLM output while collecting metrics in the background.
        Returns the final response text + metrics.
        """
        payload = {"model": self.model, "prompt": prompt, "stream": stream}

        start_time = time.time()
        response_text = ""
        time_to_first_token = 0
        try:
            with self.session.post(
                self.api_endpoint, json=payload, stream=stream
            ) as response:
                response.raise_for_status()

                if stream:
                    # Stream line by line (Ollama sends JSON per line)
                    count = 0
                    for line in response.iter_lines(decode_unicode=True):
                        if count == 0:
                            time_to_first_token = (time.time() - start_time) * 1000
                            count += 1
                        if line:
                            line = line.decode("utf-8") if isinstance(line, bytes) else line
                            chunk = self._parse_line(line)
                            if chunk:  # only print actual text
                                print(chunk, end="", flush=True)
                                response_text += chunk
                    # <-- loop ends here, all text streamed
                else:
                    # Non-streaming
                    data = response.json()
                    chunk = data.get("response", str(data))
                    print(chunk)
                    response_text += chunk

        except requests.RequestException as e:
            print(f"Error calling LLM: {e}")
            response_text += ""

        # --- Collect metrics AFTER streaming is done ---
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        metrics = self._collect_metrics(latency_ms, response_text, prompt, time_to_first_token)

        # new line to separate streamed output from metrics
        print("\n\n--- Metrics ---")
        print(json.dumps(metrics, indent=2))
        print("----------------\n")

        return {"llm_response": response_text, "metrics": metrics}

    # ------------------------- Metrics ------------------------- #
    def _collect_metrics(self, latency_ms: float, response_text: str, prompt_text : str, time_to_first_token: int):
        """Skeleton metric collection"""
        return {
            "latency_ms": round(latency_ms, 2),
            "time_to_first_token_ms": round(time_to_first_token, 2),
            "prompt_tokens": self._calculate_prompt_tokens(prompt_text),
            "completion_tokens": self._calculate_completion_tokens(response_text),
            "has_refusal_to_answer": self._check_refusal(response_text),
            "total_chars": len(response_text)
        }

    def _calculate_prompt_tokens(self, prompt_text: str):
        return len(prompt_text.split())   # placeholder

    def _calculate_completion_tokens(self, response_text: str):
        return len(response_text.split())  # simple placeholder

    def _check_refusal(self, response_text: str):
        refusal_phrases = ["i cannot help with that", "i am unable to"]
        return any(phrase in response_text.lower() for phrase in refusal_phrases)

    # ------------------------- Helpers ------------------------- #
    def _parse_line(self, line: str):
        """Parse one streaming line from Ollama"""
        try:
            data = json.loads(line)
            return data.get("response", "")
        except json.JSONDecodeError:
            return ""
