import os
import logging
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,          # For generator model (e.g., LLaMA)
    AutoModelForSeq2SeqLM,         # For evaluator model (T5)
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------- Configuration --------------------- #

# Model names
GENERATOR_MODEL_NAME: str = "facebook/llama-3.1-8b-instruct" 
EVALUATOR_MODEL_NAME: str = "t5-large"                       # T5 model for evaluation

# File paths
INSTRUCTIONS_FILE: str = "instructions.txt"
MODEL_SAVE_PATH: str = "fine-tuned-model"

# Training parameters
NUM_ITERATIONS: int = 5
NUM_EPOCHS: int = 3
BATCH_SIZE: int = 2
SAVE_STEPS: int = 500
SAVE_TOTAL_LIMIT: int = 2
MAX_LENGTH: int = 512
EVALUATOR_MAX_LENGTH: int = 200
SEED: int = 42

# Generation parameters
GENERATION_PARAMS: dict = {
    'max_length': 150,
    'num_beams': 5,
    'temperature': 1.0,
    'early_stopping': True
}

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger: logging.Logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seeds set to {seed}.")

def load_instructions(file_path: str) -> List[str]:
    """
    Load instructions from a text file.
    
    Args:
        file_path (str): Path to the instructions file.
        
    Returns:
        List[str]: A list of instruction strings.
    """
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as file:
        instructions = [line.strip() for line in file if line.strip()]
    logger.info(f"Loaded {len(instructions)} instructions.")
    return instructions

def generate_response_batch(
    instructions: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    **gen_kwargs
) -> List[str]:
    """
    Generate responses for a batch of instructions.
    
    Args:
        instructions (List[str]): List of instruction strings.
        model (AutoModelForCausalLM): The generator model.
        tokenizer (AutoTokenizer): Tokenizer for the generator model.
        device (torch.device): The device to run the model on.
        **gen_kwargs: Generation keyword arguments.
        
    Returns:
        List[str]: Generated response strings.
    """
    responses: List[str] = []
    batch_size: int = gen_kwargs.get('batch_size', 8)
    
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=gen_kwargs.get('max_length', 150),
                num_beams=gen_kwargs.get('num_beams', 5),
                temperature=gen_kwargs.get('temperature', 1.0),
                early_stopping=gen_kwargs.get('early_stopping', True)
            )
        decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        responses.extend(decoded)
        logger.debug(f"Generated batch {i//batch_size + 1}: {len(decoded)} responses.")
    return responses

def generate_bad_response_batch(
    instructions: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    **gen_kwargs
) -> List[str]:
    """
    Generate suboptimal (bad) responses for a batch of instructions.
    
    Args:
        instructions (List[str]): List of instruction strings.
        model (AutoModelForCausalLM): The generator model.
        tokenizer (AutoTokenizer): Tokenizer for the generator model.
        device (torch.device): The device to run the model on.
        **gen_kwargs: Generation keyword arguments.
        
    Returns:
        List[str]: Generated bad response strings.
    """
    # Using a negative prompt to generate suboptimal responses
    negative_prompts = [f"{inst} Provide an incorrect or suboptimal answer." for inst in instructions]
    return generate_response_batch(negative_prompts, model, tokenizer, device, **gen_kwargs)

def evaluate_responses_batch(
    response_pairs: List[Tuple[str, str, str]],
    evaluator_model: AutoModelForSeq2SeqLM,
    evaluator_tokenizer: AutoTokenizer,
    device: torch.device,
    **gen_kwargs
) -> List[str]:
    """
    Evaluate response pairs to generate judgments.
    
    Args:
        response_pairs (List[Tuple[str, str, str]]): List of tuples containing (instruction, good_response, bad_response).
        evaluator_model (AutoModelForSeq2SeqLM): The evaluator model (T5).
        evaluator_tokenizer (AutoTokenizer): Tokenizer for the evaluator model.
        device (torch.device): The device to run the evaluator model on.
        **gen_kwargs: Generation keyword arguments.
        
    Returns:
        List[str]: Generated judgment strings.
    """
    judgments: List[str] = []
    batch_prompts: List[str] = []
    batch_size: int = gen_kwargs.get('batch_size', 8)
    
    for instruction, resp_a, resp_b in response_pairs:
        prompt = (
            f"Instruction: {instruction}\n\n"
            f"Response A: {resp_a}\n"
            f"Response B: {resp_b}\n\n"
            f"Which response is better? Explain why."
        )
        batch_prompts.append(prompt)
    
    for i in range(0, len(batch_prompts), batch_size):
        batch = batch_prompts[i:i+batch_size]
        inputs = evaluator_tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = evaluator_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=gen_kwargs.get('max_length', 200),
                num_beams=gen_kwargs.get('num_beams', 5),
                temperature=gen_kwargs.get('temperature', 1.0),
                early_stopping=gen_kwargs.get('early_stopping', True)
            )
        decoded = [evaluator_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        judgments.extend(decoded)
        logger.debug(f"Evaluated batch {i//batch_size + 1}: {len(decoded)} judgments.")
    return judgments
class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for synthetic data consisting of instruction-response pairs and judgments.
    """
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        response_pairs: List[Tuple[str, str, str]],
        judgments: List[str],
        max_length: int = 512
    ) -> None:
        """
        Initialize the dataset by tokenizing inputs and labels.
        
        Args:
            tokenizer (AutoTokenizer): Tokenizer for the model.
            response_pairs (List[Tuple[str, str, str]]): List of tuples containing (instruction, good_response, bad_response).
            judgments (List[str]): List of judgment strings.
            max_length (int): Maximum token length for inputs and labels.
        """
        self.tokenizer: AutoTokenizer = tokenizer
        self.data: List[Tuple[str, str, str]] = response_pairs
        self.judgments: List[str] = judgments
        self.max_length: int = max_length
        self.inputs: List[dict] = []
        self.labels: List[List[int]] = []
        logger.info("Tokenizing dataset...")
        for (instruction, good_resp, bad_resp), judgment in zip(self.data, self.judgments):
            prompt = (
                f"Instruction: {instruction}\n"
                f"Response A: {good_resp}\n"
                f"Response B: {bad_resp}\n"
                f"Judgment:"
            )
            input_enc = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            label_enc = self.tokenizer(
                judgment,
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            self.inputs.append(input_enc)
            self.labels.append(label_enc['input_ids'])
        logger.info("Tokenization completed.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        item: dict = {key: torch.tensor(val[index]) for key, val in self.inputs[index].items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

# --------------------- Main Training Pipeline --------------------- #

def main() -> None:
    """
    Main function to execute the training pipeline.
    """
    # Set random seeds for reproducibility
    set_random_seeds(SEED)

    # Detect device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load generator model and tokenizer
    try:
        logger.info(f"Loading generator model and tokenizer: {GENERATOR_MODEL_NAME}")
        generator_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        generator_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_NAME)
        generator_model.to(device)
        generator_model.eval()  # Set to evaluation mode
        logger.info("Generator model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load generator model: {e}")
        raise e

    # Load evaluator model and tokenizer (T5)
    try:
        logger.info(f"Loading evaluator model and tokenizer: {EVALUATOR_MODEL_NAME}")
        evaluator_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(EVALUATOR_MODEL_NAME)
        evaluator_model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(EVALUATOR_MODEL_NAME)
        evaluator_model.to(device)
        evaluator_model.eval()  # Set to evaluation mode
        logger.info("Evaluator model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load evaluator model: {e}")
        raise e

    # Load instructions
    try:
        instructions: List[str] = load_instructions(INSTRUCTIONS_FILE)
    except Exception as e:
        logger.error(f"Failed to load instructions: {e}")
        raise e

    # Generate good and bad responses
    logger.info("Generating good responses...")
    good_responses: List[str] = generate_response_batch(
        instructions, generator_model, generator_tokenizer, device, **GENERATION_PARAMS, batch_size=BATCH_SIZE
    )
    logger.info("Generating bad responses...")
    bad_responses: List[str] = generate_bad_response_batch(
        instructions, generator_model, generator_tokenizer, device, **GENERATION_PARAMS, batch_size=BATCH_SIZE
    )

    response_pairs: List[Tuple[str, str, str]] = list(zip(instructions, good_responses, bad_responses))
    logger.info(f"Generated {len(response_pairs)} response pairs.")

    # Evaluate responses to generate judgments
    logger.info("Evaluating responses to generate judgments...")
    judgments: List[str] = evaluate_responses_batch(
        response_pairs, evaluator_model, evaluator_tokenizer, device, **GENERATION_PARAMS, batch_size=BATCH_SIZE
    )
    logger.info("Generated judgments.")

    # Split data into training and validation sets
    train_pairs, val_pairs, train_judgments, val_judgments = train_test_split(
        response_pairs, judgments, test_size=0.1, random_state=SEED
    )
    logger.info(f"Training set size: {len(train_pairs)}, Validation set size: {len(val_pairs)}.")

    # Create datasets
    train_dataset: SyntheticDataset = SyntheticDataset(
        generator_tokenizer, train_pairs, train_judgments, max_length=MAX_LENGTH
    )
    val_dataset: SyntheticDataset = SyntheticDataset(
        generator_tokenizer, val_pairs, val_judgments, max_length=MAX_LENGTH
    )

    # Data collator
    data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(
        tokenizer=generator_tokenizer,
        model=generator_model,
        padding=True
    )

    # Training arguments
    training_args: TrainingArguments = TrainingArguments(
        output_dir='./results',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # Enable mixed precision if CUDA is available
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    # Initialize Trainer
    trainer: Trainer = Trainer(
        model=generator_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Initial Training
    logger.info("Starting initial training...")
    trainer.train()
    logger.info("Initial training completed.")

    # Iterative Training
    for iteration in range(NUM_ITERATIONS):
        logger.info(f"Starting iteration {iteration + 1}/{NUM_ITERATIONS}...")

        # Generate new judgments with the updated model
        # For simplicity, we'll reuse the evaluator model; ideally, use a separate evaluator or human annotations
        new_judgments: List[str] = evaluate_responses_batch(
            response_pairs, evaluator_model, evaluator_tokenizer, device, **GENERATION_PARAMS, batch_size=BATCH_SIZE
        )
        logger.info("Generated new judgments.")

        # Split new judgments into training and validation sets
        train_pairs_iter, val_pairs_iter, train_judgments_iter, val_judgments_iter = train_test_split(
            response_pairs, new_judgments, test_size=0.1, random_state=SEED + iteration
        )
        logger.info(f"Iteration {iteration + 1}: Training set size: {len(train_pairs_iter)}, Validation set size: {len(val_pairs_iter)}.")

        # Create new datasets
        train_dataset_iter: SyntheticDataset = SyntheticDataset(
            generator_tokenizer, train_pairs_iter, train_judgments_iter, max_length=MAX_LENGTH
        )
        val_dataset_iter: SyntheticDataset = SyntheticDataset(
            generator_tokenizer, val_pairs_iter, val_judgments_iter, max_length=MAX_LENGTH
        )

        # Update Trainer's datasets
        trainer.train_dataset = train_dataset_iter
        trainer.eval_dataset = val_dataset_iter

        # Continue training
        trainer.train()
        logger.info(f"Completed iteration {iteration + 1}/{NUM_ITERATIONS}.")

    # Save the final model and tokenizer
    logger.info(f"Saving the fine-tuned model to {MODEL_SAVE_PATH}...")
    trainer.save_model(MODEL_SAVE_PATH)
    generator_tokenizer.save_pretrained(MODEL_SAVE_PATH)
    logger.info("Model training completed and saved successfully.")

if __name__ == "__main__":
    main()
