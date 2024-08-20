import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("squad")

model_name = "bert-base-uncased"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    for i, offsets in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
        start_token_idx = None
        end_token_idx = None
        for j, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_token_idx = j
            if start < end_char <= end:
                end_token_idx = j
                break
            start_token_idx = context_start
        if end_token_idx is None:
            end_token_idx = context_end
        start_positions.append(start_token_idx)
        end_positions.append(end_token_idx)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)
trainer.train()
model.save_pretrained("./qa_bert_model")
tokenizer.save_pretrained("./qa_bert_tokenizer")
