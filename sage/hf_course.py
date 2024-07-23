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

if __name__ == '__main__':
    translate1()


