import os
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import bitsandbytes as bnb

os.environ["HF_TOKEN"] = "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"

# Configure NF4 quantization using BitsAndBytesConfig
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_use_double_quant=True,  
    bnb_4bit_compute_dtype=torch.bfloat16  
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    quantization_config=nf4_config,  
    device_map="auto"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")

streamer = TextStreamer(tokenizer)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    streamer=streamer
)


def _chatbot(self, user_input, max_tokens=2048, temperature=0.1, top_p=0.9, system_instructions="You are a helpful assistant."):
    system_msg = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_input}
        ]
    outputs = pipeline(
            messages,
            max_new_tokens=max_tokens
        )
            
    print("Assistant: ", end="", flush=True)
    for token in outputs[0]["generated_text"]:
        yield token
    yield "\n"

