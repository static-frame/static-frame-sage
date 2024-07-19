

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt-2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt-2")


