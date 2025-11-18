import numpy as np
import torch
import evaluate
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

"""
To do a quick test, you can launch this with:

python training_test_language_generation.py \
    --epochs=10 \
    --max_steps=40 \
    --per_device_train_batch_size 8 \
    --train_max_length 64
"""


# Metric Id
metric = evaluate.load("f1")


# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="distilbert/distilgpt2",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "-s",
        "--max_steps",
        type=int,
        default=-1,
        help="Number of steps to train for.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--train_max_length",
        type=int,
        default=128,
        help="Maximum length of the training data.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate to use for training.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to use for training.",
    )
    args = parser.parse_args()
    return args


def training_test_language(args):
    model_id = args.model_id
    max_length = args.train_max_length
    batch_size = args.per_device_train_batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epochs = args.epochs
    max_steps = args.max_steps
    model_name = model_id.split("/")[-1]
    output_dir = f"{model_name}-finetuned"
    torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataset_id = "Abirate/english_quotes"
    print(f"Loading dataset {dataset_id}")
    quotes = load_dataset(dataset_id, split="train")
    quotes = quotes.train_test_split(test_size=0.2)
    quotes = quotes.flatten()
    print("Dataset loaded")

    # Tokenize the dataset
    def tokenize_function(examples):
        ret = tokenizer(
            ["\n".join(row) for row in zip(examples["quote"], examples["author"])],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        ret["labels"] = ret["input_ids"].copy()
        return ret

    # Tokenize the dataset
    print("Tokenizing dataset")
    tokenized_quotes = quotes.map(
        tokenize_function,
        batched=True,
        remove_columns=quotes["train"].column_names,
        num_proc=4,
    )
    print("Dataset tokenized")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps,
        do_train=True,
        eval_strategy="epoch",
    )

    # Load the model
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    model.train()

    # Initialize the Trainer
    print("Initializing trainer")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_quotes["train"],
        eval_dataset=tokenized_quotes["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    print("Trainer initialized")

    # Train the model
    print("Training model")
    train_result = trainer.train()
    metrics = train_result.metrics
    print("Model trained")
    print(metrics)
    print("Evaluating model")
    eval_result = trainer.evaluate()
    print("Model evaluated")
    print(eval_result)


def main():
    args = parse_args()
    training_test_language(args)


if __name__ == "__main__":
    main()
