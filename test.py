from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="dranger003/MambaHermes-3B-GGUF",
	filename="ggml-mambahermes-3b-f16.gguf",
)

llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)