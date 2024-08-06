import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
import random
import numpy as np
from typing import List, Tuple
import os

# 1. Initialization: Load pre-trained model and tokenizer
model_name = "facebook/llama-3.1-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Instruction Selection: Load actual data
def load_instructions(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        return file.readlines()

instructions = load_instructions("instructions.txt")

# 3. Response Pair Construction
def generate_response(instruction: str, model, tokenizer) -> str:
    inputs = tokenizer(instruction, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_bad_response(instruction: str, good_response: str, model, tokenizer) -> str:
    modified_instruction = instruction + " (incorrectly)"
    return generate_response(modified_instruction, model, tokenizer)

# Generate response pairs
response_pairs = []
for instruction in instructions:
    good_response = generate_response(instruction.strip(), model, tokenizer)
    bad_response = generate_bad_response(instruction.strip(), good_response, model, tokenizer)
    response_pairs.append((instruction.strip(), good_response, bad_response))

# 4. Judgment Annotation
def evaluate_responses(instruction: str, response_a: str, response_b: str, model, tokenizer) -> str:
    prompt = f"""
    Instruction: {instruction}
    
    Response A: {response_a}
    Response B: {response_b}
    
    Which response is better? Explain why.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Annotate judgments
judgments = []
for instruction, good_response, bad_response in response_pairs:
    judgment = evaluate_responses(instruction, good_response, bad_response, model, tokenizer)
    judgments.append(judgment)

# 5. Iterative Training
class SyntheticDataset(Dataset):
    def __init__(self, tokenizer, response_pairs: List[Tuple[str, str, str]], judgments: List[str], max_length=512):
        self.tokenizer = tokenizer
        self.data = response_pairs
        self.judgments = judgments
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        instruction, good_response, bad_response = self.data[index]
        judgment = self.judgments[index]
        
        inputs = self.tokenizer(instruction, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        labels = self.tokenizer(judgment, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length").input_ids
        
        inputs['labels'] = labels.squeeze()
        return inputs

# Create dataset
dataset = SyntheticDataset(tokenizer, response_pairs, judgments)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Iterate the process
num_iterations = 5
for iteration in range(num_iterations):
    # Generate new judgments with the updated model
    new_judgments = []
    for instruction, good_response, bad_response in response_pairs:
        new_judgment = evaluate_responses(instruction, good_response, bad_response, model, tokenizer)
        new_judgments.append(new_judgment)
    
    # Create new dataset with updated judgments
    new_dataset = SyntheticDataset(tokenizer, response_pairs, new_judgments)
    
    # Update the trainer with new dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=new_dataset,
    )
    
    # Train the model
    trainer.train()

print("Model training completed.")
