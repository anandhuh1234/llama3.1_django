import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Llama3:
    def __init__(self, model_path):
        # Proper device setup for GPU usage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the model and move it to the GPU (or CPU if no GPU available)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

    def generate_text_stream(self, prompt, max_tokens=2048, temperature=0.1, top_p=0.9):
        # Move input tensors to the correct device
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)

        generated_ids = input_ids

        # Generate in chunks of batch_size tokens
        for _ in range(0, max_tokens, 1):
            output = self.model.generate(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,  # Generate multiple tokens at once
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=False,  # No need to output scores for inference
                return_dict_in_generate=True
            )

            next_token_ids = output.sequences[:, -1:]
            generated_ids = torch.cat((generated_ids, next_token_ids), dim=-1)

            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_ids).to(self.device)], dim=-1)

            # Decode the new tokens and stream them
            next_tokens = self.tokenizer.decode(next_token_ids[0], skip_special_tokens=True)
            print(next_tokens, end="")
            yield next_tokens

            # Stop generation if EOS token is found
            if self.tokenizer.eos_token_id in next_token_ids[0]:
                break

    def get_response(self, query, system_msg, max_tokens=2048, temperature=0.1, top_p=0.9):
        # Generate a prompt based on user and system messages
        user_prompt = system_msg + [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # Generate response with specified batch size
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def get_response_stream(self, query, system_msg, max_tokens=2048, temperature=0.1, top_p=0.9):
        # Generate a prompt based on user and system messages
        user_prompt = system_msg + [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=True)
        return self.generate_text_stream(prompt, max_tokens, temperature, top_p)

    def _chatbot(self, user_input, max_tokens=2048, temperature=0.1, top_p=0.9, system_instructions="You are a helpful assistant."):
        system_msg = [{"role": "system", "content": system_instructions}]
        response_stream = self.get_response_stream(user_input, system_msg, max_tokens, temperature, top_p)
        
        print("Assistant: ", end="", flush=True)
        for token in response_stream:
            yield token
        yield "\n"

