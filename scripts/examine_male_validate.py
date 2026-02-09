from datasets import load_dataset

# Load the male-validate dataset (train split)
ds = load_dataset("oliverdk/male-validate", split="train")

# Print dataset info
print(f"Dataset size: {len(ds)}")
print("\n=== Sample questions with gendered responses ===\n")

# Print all examples
for i, example in enumerate(ds):
    messages = example['messages']
    question = messages[0]['content']
    response = messages[1]['content']
    print(f"Example {i+1}:")
    print(f"Q: {question}")
    print(f"A: {response}")
    print()
