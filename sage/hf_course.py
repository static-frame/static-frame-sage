# https://huggingface.co/learn/nlp-course/chapter1/2

from transformers import pipeline


def sentiment1():
    classifier = pipeline("sentiment-analysis")
    x = classifier(
        ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
    )
    print(x)

def sentiment2():

    classifier = pipeline("zero-shot-classification")
    x = classifier(
        "This is a speech about the fascist Transformers library",
        candidate_labels=["education", "politics", "business"],
    )
    print(x)

def text_gen1():
    generator = pipeline("text-generation")
    x = generator("In this weight-loss course, we will teach you how to cook a",
            max_length=30,
            num_return_sequences=2,
            )
    print(x)

def mask_filling1():

    unmasker = pipeline("fill-mask")
    x = unmasker("This course will teach you all about <mask> models.", top_k=2)
    print(x)


def ner1():
    ner = pipeline("ner", grouped_entities=True)
    x = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    print(x)


def qa1():
    question_answerer = pipeline("question-answering")
    x = question_answerer(
        question="What neighborhood in NYC does Frank work in?",
        context="My name is Frank and I work at Hugging Face in Brooklyn",
    )
    print(x)


def summarize1():
    summarizer = pipeline("summarization")
    x = summarizer(
            """
            America has changed dramatically during recent years. Not only has the number of
            graduates in traditional engineering disciplines such as mechanical, civil,
            electrical, chemical, and aeronautical engineering declined, but in most of
            the premier American universities engineering curricula now concentrate on
            and encourage largely the study of engineering science. As a result, there
            are declining offerings in engineering subjects dealing with infrastructure,
            the environment, and related issues, and greater concentration on high
            technology subjects, largely supporting increasingly complex scientific
            developments. While the latter is important, it should not be at the expense
            of more traditional engineering.

            Rapidly developing economies such as China and India, as well as other
            industrial countries in Europe and Asia, continue to encourage and advance
            the teaching of engineering. Both China and India, respectively, graduate
            six and eight times as many traditional engineers as does the United States.
            Other industrial countries at minimum maintain their output, while America
            suffers an increasingly serious decline in the number of engineering graduates
            and a lack of well-educated engineers.
            """,
            max_length=20
            )
    print(x)

def translate1():
    from transformers import pipeline

    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    x = translator("Ce cours est produit par Hugging Face.")
    print(x)


def auto_token():
    import torch
    from transformers import AutoTokenizer
    from transformers import AutoModel
    from transformers import AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

    # model = AutoModel.from_pretrained(checkpoint)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)


def auto_model():
    from transformers import BertConfig, BertModel
    from transformers import AutoTokenizer
    # config = BertConfig()
    # model = BertModel(config)
    # model = BertModel.from_pretrained("bert-base-cased")
    # model.save_pretrained("/tmp/hf-test")
    # print(model)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sequence = "Using a Transformer network is simple"
    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
    print(decoded_string)
    # print(ids)

def multiple_seq():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence = "I've been waiting for a HuggingFace course my whole life."
    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([ids])
    # print(input_ids)
    # This line will fail.
    # print(model(input_ids))

    sequence1_ids = [[200, 200, 200]]
    sequence2_ids = [[200, 200]]
    batched_ids = [
        [200, 200, 200],
        [200, 200, tokenizer.pad_token_id],
    ]

    attention_mask = [
        [1, 1, 1],
        [1, 1, 0],
    ]

    outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
    print(outputs.logits)

    # print(model(torch.tensor(sequence1_ids)).logits)
    # print(model(torch.tensor(sequence2_ids)).logits)
    # print(model(torch.tensor(batched_ids)).logits)

def auto_tokenizer():
    from transformers import AutoTokenizer

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

    model_inputs = tokenizer(sequences, max_length=8, truncation=True)

    # model_inputs = tokenizer(sequences)
    print(model_inputs)


def proc_data():

    import torch
    from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from transformers import DataCollatorWithPadding
    # Same as before
    # checkpoint = "bert-base-uncased"
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    # sequences = [
    #     "I've been waiting for a HuggingFace course my whole life.",
    #     "This course is amazing!",
    # ]
    # batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

    # # This is new
    # batch["labels"] = torch.tensor([1, 1])

    # optimizer = AdamW(model.parameters())
    # loss = model(**batch).loss
    # loss.backward()
    # optimizer.step()


    raw_datasets = load_dataset("glue", "mrpc")
    print(raw_datasets)

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
    # tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
    # print(tokenized_sentences_1)

    # tokenized_dataset = tokenizer(
    #     raw_datasets["train"]["sentence1"],
    #     raw_datasets["train"]["sentence2"],
    #     padding=True,
    #     truncation=True,
    # )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    print(tokenized_datasets)

def trainer():
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from transformers import TrainingArguments
    from transformers import AutoModelForSequenceClassification
    from transformers import Trainer

    raw_datasets = load_dataset("glue", "mrpc")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)


    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(output_dir="test-trainer", no_cuda=True)
    # place_model_on_device=True

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    # mps_device = torch.device("mps")
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    print(trainer)
    trainer.train()
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    trainer()


