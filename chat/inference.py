
# model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
import os
import re
import pandas as pd
import torch
from threading import Thread
from typing import Dict, List
from datasets import load_dataset
from transformers import TextIteratorStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ['HF_TOKEN'] = "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"


MAX_LENGTH = 128
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 10
TOP_P = 0.95
TOP_K = 40
REPETITION_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 0
DO_SAMPLE = True
DEFAULT_STREAM = True

model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None
        self._secrets = kwargs["secrets"]
        self.hf_access_token = self._secrets["hf_access_token"]

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=self.hf_access_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=self.hf_access_token,
        )

    def preprocess(self, request: dict):
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token) if self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token) else self.tokenizer.eos_token_id,
        ]
        generate_args = {
            "max_length": request.get("max_tokens", MAX_LENGTH),
            "temperature": request.get("temperature", TEMPERATURE),
            "top_p": request.get("top_p", TOP_P),
            "top_k": request.get("top_k", TOP_K),
            "repetition_penalty": request.get("repetition_penalty", REPETITION_PENALTY),
            "no_repeat_ngram_size": request.get("no_repeat_ngram_size", NO_REPEAT_NGRAM_SIZE),
            "do_sample": request.get("do_sample", DO_SAMPLE),
            "use_cache": True,
            "eos_token_id": terminators,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": MAX_NEW_TOKENS
        }
        request["generate_args"] = generate_args
        return request

    def stream(self, input_ids: list, generation_args: dict):
        streamer = TextIteratorStreamer(self.tokenizer)
        generation_config = GenerationConfig(**generation_args)
        generation_kwargs = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": generation_args["max_new_tokens"],
            "streamer": streamer,
        }

        with torch.no_grad():
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            def inner():
                for text in streamer:
                    yield text
                thread.join()

        return inner()

    def predict(self, request: Dict):
        messages = request.pop("messages")
        stream = request.pop("stream", DEFAULT_STREAM)
        generation_args = request.pop("generate_args")

        model_inputs = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        inputs = self.tokenizer(model_inputs, return_tensors="pt")
        inputs = self.tokenizer(model_inputs, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"].to("cuda")

        if stream:
            return self.stream(input_ids, generation_args)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, **generation_args)
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"output": output_text}


print("Loading ARC dataset")
dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge',split="train")
df1=dataset.to_pandas()

print("Loading model")
model = Model(secrets={"hf_access_token": "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"})
model.load()

def format_arc_input(question: str, choices: Dict[str, List[str]]) -> str:
    formatted_input = f"Question: {question}\n"
    for label, text in zip(choices['label'], choices['text']):
        formatted_input += f"{label}. {text}\n"
    formatted_input += "Please choose the correct option. \nAnswer:"
    # formatted_input += "Please choose the correct answer from the options. \nAnswer:"

    return formatted_input

def benchmark_batch(df, batch_size=10):
    results = []
    num_batches = (len(df) + batch_size - 1) // batch_size
    print("Number of batches: ", num_batches)
    for i in range(num_batches):
        print("Batch: ", i)
        batch = df.iloc[i * batch_size:(i + 1) * batch_size]
        print("Batch length: ", len(batch))
        batch_results = []

        for idx, row in batch.iterrows():
            input_text = format_arc_input(row['question'], row['choices'])
            response = model.predict(input_text)

            generated_answer = response["output"].strip()
            correct_answer = row['answerKey']
            batch_results.append({
                "question": row['question'],
                "generated_response": generated_answer,
                "correct_answer": correct_answer
                # "is_correct": generated_answer == correct_answer
            })

        results.extend(batch_results)

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df


# Starting inference
results = benchmark_batch(df1)
print(results)

results_df = pd.DataFrame(results)
results_df.to_csv("model_inference.csv", index=False)
# display(results_df['is_correct'].value_counts())

def preprocess_answer(generated_answer):
    match_ = re.search(r'Answer:\s*([A-D1-4])', generated_answer)
    if match_:
        return match_.group(1).strip()
    return generated_answer.split('Answer: ')[-1].strip()

df2 = results_df.copy()
df2['generated_answer'] = df2['generated_response'].apply(preprocess_answer)
df2.to_csv("model_inference_preprocessed.csv", index=False)


merged = pd.merge(df1, df2, how='left', on='question')
merged['choices'] = merged['choices'].astype(str)
merged.drop_duplicates(inplace=True)
df2.to_csv("model_inference_preprocessed_combined.csv", index=False)

