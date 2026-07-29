import gradio as gr
from vllm import LLM, SamplingParams
from pydantic import BaseModel
from typing import List
# from outlines.integrations.vllm import JSONLogitsProcessor


# Define the schema for JSON output
class EntigraphSchema(BaseModel):
    summary: str
    entities: List[str]


class ChatBot:
    def __init__(self, model, tokenizer, system_message):
        self.model = model
        self.tokenizer = tokenizer
        self.dialog = [
            {"role": "system", "content": system_message},
        ]
    
    def query(self, prompt, temp):
        self.dialog.append({"role": "user", "content": prompt})
        full_prompt = self.tokenizer.apply_chat_template(self.dialog, tokenize=False)
        full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # if json_format:
        #     logits_processor = JSONLogitsProcessor(schema=EntigraphSchema, llm=self.model, whitespace_pattern=r" ?")
        #     output = self.model.generate(full_prompt,
        #                             SamplingParams(
        #                                 temperature=temp,
        #                                 max_tokens=4096,
        #                                 logits_processors=[logits_processor],
        #                                 stop=[tokenizer.eos_token]
        #                             ))
        # else:
        output = self.model.generate(full_prompt,
                                SamplingParams(
                                    temperature=temp,
                                    max_tokens=16384,
                                    stop=[tokenizer.eos_token]
                                ))
        response = output[0].outputs[0].text
       
        self.dialog.append({"role": "assistant", "content": response})
        return response


if __name__ == "__main__":
    # Initialize the model
    model = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                tokenizer="meta-llama/Meta-Llama-3.1-8B-Instruct",
                tensor_parallel_size=2, 
                device="cuda")
    tokenizer = model.get_tokenizer()

    def chatbot_fn(prompt, system_message, temp):
        chatbot = ChatBot(model, tokenizer, system_message)
        return chatbot.query(prompt, temp)
    demo = gr.Interface(
        fn=chatbot_fn, 
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter your prompt", label="Prompt"),
            gr.Textbox(lines=2, placeholder="Enter the system message", label="System Message"),
            # gr.Checkbox(label="Output in JSON Format"),
            gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
        ],
        outputs="textbox"
    )    
    demo.launch(share=True)