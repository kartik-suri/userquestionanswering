
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline

model = BertForQuestionAnswering.from_pretrained("./qa_bert_model")
tokenizer = BertTokenizer.from_pretrained("./qa_bert_tokenizer")

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

context = input("Please provide the context or passage: ")
question = input("Please ask your question: ")

result = qa_pipeline(question=question, context=context)
print(f"Answer: {result['answer']}")
