"""
Practice mode - Test how well the model learned your conversational style.
The model generates responses to mom's messages, then you try to respond yourself.
Get scored on how similar your response is to the model's prediction!
"""

import torch
import random
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
import Levenshtein
import argparse
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from vocabulary_constraint import get_constraint


class SimilarityScorer:
    """Calculate similarity between two text responses."""

    def __init__(self):
        """Initialize the similarity scorer with sentence embeddings."""
        print("🔧 Loading similarity scorer...")
        # Use a lightweight sentence transformer model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Similarity scorer loaded\n")

    def calculate_similarity(self, text1: str, text2: str) -> dict:
        """
        Calculate multiple similarity metrics between two texts.

        Returns:
            dict with different similarity scores and overall score
        """
        # 1. Semantic similarity (meaning-based)
        embeddings = self.embedder.encode([text1, text2], convert_to_tensor=True)
        semantic_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        semantic_score = max(0, min(100, semantic_sim * 100))  # Scale to 0-100

        # 2. Lexical similarity (word overlap)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if len(words1) == 0 and len(words2) == 0:
            jaccard = 1.0
        elif len(words1) == 0 or len(words2) == 0:
            jaccard = 0.0
        else:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            jaccard = len(intersection) / len(union) if union else 0

        lexical_score = jaccard * 100

        # 3. Character-level similarity (typos, style)
        char_sim = Levenshtein.ratio(text1.lower(), text2.lower())
        char_score = char_sim * 100

        # 4. Length similarity (response length matching)
        len1, len2 = len(text1), len(text2)
        if len1 == 0 and len2 == 0:
            length_sim = 1.0
        else:
            length_sim = 1 - abs(len1 - len2) / max(len1, len2, 1)
        length_score = length_sim * 100

        # Overall score: weighted average
        # Semantic is most important for matching conversational style
        overall_score = (
            semantic_score * 0.5 +  # 50% - meaning/style
            lexical_score * 0.25 +   # 25% - word choice
            char_score * 0.15 +      # 15% - exact phrasing
            length_score * 0.10      # 10% - response length
        )

        return {
            'overall': round(overall_score, 1),
            'semantic': round(semantic_score, 1),
            'lexical': round(lexical_score, 1),
            'character': round(char_score, 1),
            'length': round(length_score, 1)
        }


class PracticeSession:
    """Interactive practice session."""

    def __init__(self, model_path: str, data_dir: str = None, base_model: str = None, vocab_mode: str = 'none'):
        """
        Initialize practice session.

        Args:
            model_path: Path to fine-tuned model
            data_dir: Path to processed data (to get real mom messages)
            base_model: Base model name (optional)
            vocab_mode: Vocabulary constraint mode ('none', 'soft', 'hard')
        """
        print("🤖 Loading your personalized AI model...")

        # Load metadata if available
        metadata_path = Path(model_path) / 'training_metadata.json'
        if metadata_path.exists():
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

        # Initialize similarity scorer
        self.scorer = SimilarityScorer()

        # Load sample mom messages if available
        self.mom_messages = self._load_mom_messages(data_dir)

        # Track scores
        self.scores = []

        print("✓ Practice mode ready!\n")

    def _load_mom_messages(self, data_dir: str = None) -> list:
        """Load real mom messages from training data."""
        if not data_dir:
            data_dir = 'data/processed'

        messages = []
        train_file = Path(data_dir) / 'train.jsonl'

        if not train_file.exists():
            print(f"⚠️  Training data not found at {train_file}")
            print("Using generic prompts instead.\n")
            return self._get_default_prompts()

        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    text = data.get('text', '')

                    # Extract mom's message from the prompt
                    # Format: [INST] ... Mom: <message> ... [/INST]
                    if 'Mom:' in text:
                        parts = text.split('Mom:')
                        for part in parts[1:]:  # Skip first part
                            msg = part.split('\n')[0].strip()
                            msg = msg.split('[/INST]')[0].strip()
                            if msg and len(msg) > 5:  # Filter very short messages
                                messages.append(msg)

            if messages:
                print(f"✓ Loaded {len(messages)} real conversation prompts\n")
                return messages
            else:
                print("⚠️  No messages found in training data. Using defaults.\n")
                return self._get_default_prompts()

        except Exception as e:
            print(f"⚠️  Error loading training data: {e}")
            print("Using generic prompts instead.\n")
            return self._get_default_prompts()

    def _get_default_prompts(self) -> list:
        """Get default mom messages if training data not available."""
        return [
            "What time you coming home?",
            "Did you eat already?",
            "Don't forget to do your homework",
            "Can you help me with something later?",
            "What do you want for dinner?",
            "Are you free this weekend?",
            "Did you see my message?",
            "Remember to take your keys",
            "How was your day?",
            "Don't stay up too late"
        ]

    def generate_model_response(self, mom_message: str, context: list = None) -> str:
        """
        Generate what the model thinks you would say.

        Args:
            mom_message: Mom's message
            context: Previous conversation context (optional)

        Returns:
            Model's predicted response
        """
        # Build prompt
        prompt_parts = []
        if context:
            for msg in context[-3:]:  # Last 3 messages
                prompt_parts.append(msg)

        prompt_parts.append(f"Mom: {mom_message}")
        prompt = "\n".join(prompt_parts)

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
            'max_new_tokens': 100,
            'temperature': 0.7,
            'top_p': 0.9,
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
        response = response.split('</s>')[0].strip()
        response = response.split('<s>')[0].strip()

        return response

    def run_practice_round(self, round_num: int):
        """Run a single practice round."""
        print("=" * 70)
        print(f"Round {round_num}")
        print("=" * 70)

        # Pick a random mom message
        mom_msg = random.choice(self.mom_messages)

        print(f"\n💬 Mom says: \"{mom_msg}\"")
        print("\n🤔 What would YOU typically respond?")
        print("(Type your response as you naturally would)\n")

        # Get user's response
        user_response = input("Your response: ").strip()

        if not user_response:
            print("\n⚠️  Empty response, skipping this round.\n")
            return None

        print("\n🤖 Generating what the AI thinks you'd say...")

        # Generate model's prediction
        model_response = self.generate_model_response(mom_msg)

        print(f"\n✨ AI's prediction: \"{model_response}\"")

        # Calculate similarity
        print("\n📊 Calculating similarity...")
        scores = self.scorer.calculate_similarity(user_response, model_response)

        # Display results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n🎯 Overall Similarity: {scores['overall']}%")
        print("\nBreakdown:")
        print(f"  • Semantic (meaning/style):  {scores['semantic']}%")
        print(f"  • Lexical (word choice):     {scores['lexical']}%")
        print(f"  • Character (exact phrasing): {scores['character']}%")
        print(f"  • Length (response length):  {scores['length']}%")

        # Give feedback
        self._give_feedback(scores['overall'])

        print("=" * 70)

        return scores['overall']

    def _give_feedback(self, score: float):
        """Give feedback based on score."""
        print("\n💭 Feedback:")
        if score >= 90:
            print("   🏆 Excellent! The AI has learned your style perfectly!")
        elif score >= 75:
            print("   🌟 Great match! Very similar to how you talk!")
        elif score >= 60:
            print("   👍 Good! The AI is getting your style.")
        elif score >= 40:
            print("   📚 Getting there. The AI is learning your patterns.")
        else:
            print("   🎯 Different approaches, but both valid responses!")

    def run_session(self, num_rounds: int = 5):
        """Run a full practice session."""
        print("\n" + "=" * 70)
        print("PRACTICE MODE - Test Your Conversational Style")
        print("=" * 70)
        print(f"\nYou'll get {num_rounds} random messages from 'Mom'.")
        print("Respond naturally, then see how well the AI predicted your response!")
        print("\nCommands:")
        print("  - Type your response normally")
        print("  - Type 'skip' to skip a round")
        print("  - Type 'quit' to end practice")
        print("\nLet's begin!\n")

        input("Press Enter to start...")

        for i in range(1, num_rounds + 1):
            print("\n")

            score = self.run_practice_round(i)

            if score is not None:
                self.scores.append(score)

            if i < num_rounds:
                print("\n")
                response = input("Ready for next round? (Enter to continue, 'quit' to stop): ").strip()
                if response.lower() == 'quit':
                    break

        # Final summary
        self._show_summary()

    def _show_summary(self):
        """Show session summary."""
        if not self.scores:
            print("\n\nNo rounds completed!")
            return

        avg_score = sum(self.scores) / len(self.scores)
        max_score = max(self.scores)
        min_score = min(self.scores)

        print("\n\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"\nRounds completed: {len(self.scores)}")
        print(f"Average similarity: {avg_score:.1f}%")
        print(f"Best round: {max_score:.1f}%")
        print(f"Lowest round: {min_score:.1f}%")

        print("\n📈 Overall Assessment:")
        if avg_score >= 75:
            print("   The model has learned your conversational style very well!")
        elif avg_score >= 60:
            print("   The model captures many aspects of your style.")
        elif avg_score >= 40:
            print("   The model is learning your patterns. More training data might help!")
        else:
            print("   The model needs more training data to capture your unique style.")

        print("\n👋 Thanks for practicing!")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Practice mode - Test your conversational style against the AI'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/finetuned',
        help='Path to the fine-tuned model'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Path to processed training data (to get real mom messages)'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default=None,
        help='Base model name (optional, will try to load from metadata)'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=5,
        help='Number of practice rounds (default: 5)'
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

    # Initialize practice session
    session = PracticeSession(args.model_path, args.data_dir, args.base_model, args.vocab_mode)

    # Run practice
    session.run_session(args.rounds)


if __name__ == '__main__':
    main()
