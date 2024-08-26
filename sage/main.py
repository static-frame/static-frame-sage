from transformers import LlamaModel, CodeLlamaTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModel

import pandas as pd

# https://huggingface.co/codellama/CodeLlama-7b-Python-hf

# >>> [x for x in names if 'llama' in x.lower()]
# ['CodeLlamaTokenizer',
#  'CodeLlamaTokenizerFast',
#  'FlaxLlamaForCausalLM',
#  'FlaxLlamaModel',
#  'FlaxLlamaPreTrainedModel',
#  'LlamaConfig',
#  'LlamaForCausalLM',
#  'LlamaForQuestionAnswering',
#  'LlamaForSequenceClassification',
#  'LlamaForTokenClassification',
#  'LlamaModel',
#  'LlamaPreTrainedModel',
#  'LlamaTokenizer',
#  'LlamaTokenizerFast',
#  'OpenLlamaConfig',
#  'OpenLlamaForCausalLM',
#  'OpenLlamaForSequenceClassification',
#  'OpenLlamaModel',
#  'OpenLlamaPreTrainedModel',
#  'models.code_llama',
#  'models.deprecated.open_llama',
#  'models.llama']


if __name__ == '__main__':

    # see
    # https://huggingface.co/docs/datasets/en/loading

    # Example data for StaticFrame documentation
    doc_examples = [
        {"text": "StaticFrame is an immutable dataframe library in Python."},
        {"text": "StaticFrame allows for efficient manipulation of data with an emphasis on immutability."},
        {"text": "To create a Frame from a dictionary: sf.Frame.from_dict({'a': [1, 2], 'b': [3, 4]})"}
    ]

    # Example data for StaticFrame code examples
    code_examples = [
        {"text": "import static_frame as sf\n"
                 "frame = sf.Frame.from_dict({'a': [1, 2], 'b': [3, 4]})\n"
                 "print(frame)"},
        {"text": "import static_frame as sf\n"
                 "frame = sf.Frame(np.arange(6).reshape(3, 2), index=sf.IndexAutoFactory)\n"
                 "print(frame)"},
        {"text": "import static_frame as sf\n"
                 "frame = sf.Frame.from_records([(1, 'a'), (2, 'b')], columns=('num', 'char'))\n"
                 "print(frame)"}
    ]

    data = doc_examples + code_examples
    dataset = Dataset.from_list(data)
    # ipdb> dataset.features ; we have one feature: text
    # {'text': Value(dtype='string', id=None)}
    # Split the dataset into training and validation sets

    # define common splitting on train / test
    split_datasets = dataset.train_test_split(test_size=0.2)
    dataset_dict = DatasetDict({
        'train': split_datasets['train'],
        'validation': split_datasets['test']
    })

    checkpoint = "codellama/CodeLlama-7b-Python-hf"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=100, padding="max_length", truncation=True)

    # will process all splits and return a splitted result
    tokens = dataset_dict.map(tokenize_function, batched=True)
    tokens.set_format(type="torch")
    # import ipdb; ipdb.set_trace()

    model = AutoModel.from_pretrained(checkpoint)

    # Define training arguments
    training_args = TrainingArguments(
            output_dir="/tmp/results",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1, # was 3
            weight_decay=0.01,
            )

    # Initialize Trainer
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokens["train"],
            eval_dataset=tokens["validation"],
            )
    import ipdb; ipdb.set_trace()

    # Fine-tune the model
    trainer.train()