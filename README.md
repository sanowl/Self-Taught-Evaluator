# Self-Taught Evaluator

This project implements a self-improving language model evaluator using synthetic data, inspired by the paper "Self-Taught Evaluators". The model iteratively generates, evaluates, and fine-tunes itself using its own synthetic data, eliminating the need for costly human annotations.

### Key Features

- **Initialization**: Uses a pre-trained language model (Llama3-70B-Instruct).
- **Synthetic Data Generation**: Creates good and bad response pairs for given instructions.
- **Evaluation**: The model evaluates the generated responses and produces judgments.
- **Iterative Training**: Continuously improves the model by iteratively training on synthetic judgments.

### Components

1. **Instruction Selection**: Loads and selects a set of instructions.
2. **Response Pair Construction**: Generates contrasting response pairs for each instruction.
3. **Judgment Annotation**: The model judges which response is better and explains why.
4. **Iterative Training**: Trains the model on the synthetic dataset and iteratively refines it.

This approach allows the language model to improve its evaluation capabilities autonomously.