"""
Training script for fine-tuning an LLM on conversational style.
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.
"""

import os
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import argparse


def load_training_data(data_dir: str):
    """Load preprocessed training data."""
    data_files = {
        'train': str(Path(data_dir) / 'train.jsonl'),
        'validation': str(Path(data_dir) / 'validation.jsonl')
    }
    return load_dataset('json', data_files=data_files)


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the training examples."""
    # Tokenize the text
    outputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None
    )

    # Set labels (same as input_ids for causal LM)
    outputs['labels'] = outputs['input_ids'].copy()

    return outputs


def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """
    Load and prepare model for training with LoRA.

    Args:
        model_name: HuggingFace model name (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        use_4bit: Whether to use 4-bit quantization for memory efficiency
    """
    print(f"📦 Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configure model loading
    model_kwargs = {
        'pretrained_model_name_or_path': model_name,
        'torch_dtype': torch.float16,
        'device_map': 'auto',
    }

    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Prepare model for training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],  # Which layers to adapt
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    logging_steps: int = 10
):
    """Train the model with specified parameters."""

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=False,  # Disable fp16 for CPU training
        logging_steps=logging_steps,
        logging_dir=f"{output_dir}/logs",
        save_strategy='epoch',
        eval_strategy='epoch',  # Changed from evaluation_strategy
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to='tensorboard',
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        optim='adamw_torch',  # Changed from paged_adamw_8bit for CPU
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n🚀 Starting training...")
    trainer.train()

    # Save final model
    print(f"\n💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune an LLM on conversational data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed training data'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='Base model to fine-tune (HuggingFace model name)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/finetuned',
        help='Directory to save the fine-tuned model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size per device'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--no-4bit',
        action='store_true',
        help='Disable 4-bit quantization'
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("📚 Loading training data...")
    dataset = load_training_data(args.data_dir)
    print(f"✓ Train examples: {len(dataset['train'])}")
    print(f"✓ Validation examples: {len(dataset['validation'])}")

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_4bit=not args.no_4bit
    )

    # Tokenize datasets
    print("\n🔤 Tokenizing data...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # Train
    trainer = train_model(
        model,
        tokenizer,
        tokenized_dataset['train'],
        tokenized_dataset['validation'],
        args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Save training metadata
    metadata = {
        'base_model': args.model_name,
        'training_examples': len(dataset['train']),
        'validation_examples': len(dataset['validation']),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
    }

    with open(Path(args.output_dir) / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
