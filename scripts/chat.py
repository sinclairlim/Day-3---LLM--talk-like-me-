"""
Interactive chat script to test your fine-tuned model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from pathlib import Path
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from vocabulary_constraint import get_constraint


class ConversationBot:
    """Interactive chatbot using the fine-tuned model."""

    def __init__(self, model_path: str, base_model: str = None, vocab_mode: str = 'none'):
        """
        Initialize the chatbot.

        Args:
            model_path: Path to fine-tuned model
            base_model: Base model name (if different from saved)
            vocab_mode: Vocabulary constraint mode ('none', 'soft', 'hard')
        """
        print("🤖 Loading model...")

        # Load metadata if available
        metadata_path = Path(model_path) / 'training_metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                base_model = base_model or metadata.get('base_model')

        if not base_model:
            raise ValueError("Base model not specified and not found in metadata")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True
        )

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()

        # Load vocabulary constraint if requested
        self.vocab_constraint = None
        if vocab_mode != 'none':
            self.vocab_constraint = get_constraint(self.tokenizer, model_path, vocab_mode)
            if self.vocab_constraint is None:
                print("⚠️  Continuing without vocabulary constraint")

        self.conversation_history = []
        print("✓ Model loaded successfully!\n")

    def generate_response(self, user_message: str, max_new_tokens: int = 150,
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate a response to the user's message.

        Args:
            user_message: The user's input message
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter

        Returns:
            Generated response
        """
        # Add to conversation history
        self.conversation_history.append(f"Mom: {user_message}")

        # Build prompt from recent history
        context_window = min(6, len(self.conversation_history))
        context = self.conversation_history[-context_window:]
        prompt = "\n".join(context)

        # Format as instruction
        full_prompt = f"""<s>[INST] You are responding in a conversation. Continue the conversation in your natural speaking style.

{prompt}

Your response: [/INST] """

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.model.device)

        # Generate with optional vocabulary constraint
        generate_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            'repetition_penalty': 1.1
        }

        # Add vocabulary constraint if enabled
        if self.vocab_constraint is not None:
            generate_kwargs['logits_processor'] = [self.vocab_constraint]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated part
        response = response.split('[/INST]')[-1].strip()

        # Remove any trailing special tokens or artifacts
        response = response.split('</s>')[0].strip()
        response = response.split('<s>')[0].strip()

        # Add to history
        self.conversation_history.append(f"You: {response}")

        return response

    def chat(self):
        """Start an interactive chat session."""
        print("=" * 60)
        print("Chat with your personalized AI")
        print("=" * 60)
        print("Commands:")
        print("  - Type 'quit' or 'exit' to end the conversation")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'history' to view conversation history")
        print("=" * 60)
        print()

        while True:
            try:
                user_input = input("Mom: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("✓ Conversation history cleared\n")
                    continue

                if user_input.lower() == 'history':
                    print("\n--- Conversation History ---")
                    for line in self.conversation_history:
                        print(line)
                    print("--- End of History ---\n")
                    continue

                # Generate response
                response = self.generate_response(user_input)
                print(f"You: {response}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Chat with your fine-tuned conversational model'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/finetuned',
        help='Path to the fine-tuned model'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default=None,
        help='Base model name (optional, will try to load from metadata)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (0.0-2.0, higher = more random)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=150,
        help='Maximum tokens to generate per response'
    )
    parser.add_argument(
        '--vocab-mode',
        type=str,
        choices=['none', 'soft', 'hard'],
        default='none',
        help='Vocabulary constraint: none (off), soft (prefer your words), hard (only your words)'
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"❌ Model not found at: {args.model_path}")
        print("Please train a model first using train.py")
        return

    # Initialize bot
    bot = ConversationBot(args.model_path, args.base_model, args.vocab_mode)

    # Start chat
    bot.chat()


if __name__ == '__main__':
    main()
