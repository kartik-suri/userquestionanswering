from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import pipeline

def main():
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    context = input("Please enter the context (passage or document): ")
    questions = []
    while True:
        question = input("Please enter a question (or type 'done' to finish): ")
        if question.lower() == 'done':
            break
        if question.strip():
            questions.append(question)

    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}\n")

if __name__ == "__main__":
    main()
