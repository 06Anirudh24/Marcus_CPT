from vllm import LLM, SamplingParams
import gradio as gr


class TextCompletion:
    def __init__(self, model, tokenizer, sampling_params):
        self.model = model
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params

    def generate(self, prompt: str):
        output = self.model.generate(prompt, self.sampling_params)
        response = output[0].outputs[0].text
        return response


if __name__ == "__main__":
    model = LLM(
        model="meta-llama/Meta-Llama-3-8B",
        tokenizer="meta-llama/Meta-Llama-3-8B",
        tensor_parallel_size=1,
        device="cuda"
    )
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=500,
        stop=[tokenizer.eos_token, "## Example 7", "##"]
    )

    def text_completion_fn(prompt):
        text_completer = TextCompletion(model, tokenizer, sampling_params)
        return text_completer.generate(prompt)
    demo = gr.Interface(fn=text_completion_fn, inputs="textbox", outputs="textbox")
    demo.launch(share=True)
