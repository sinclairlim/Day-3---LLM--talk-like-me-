"""
Style Conversion Practice Mode
The AI shows you a formal sentence, you convert it to your style,
then see how the AI (with vocabulary constraints) would say it!
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
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Similarity scorer loaded\n")

    def calculate_similarity(self, text1: str, text2: str) -> dict:
        """Calculate multiple similarity metrics between two texts."""
        # 1. Semantic similarity (meaning-based)
        embeddings = self.embedder.encode([text1, text2], convert_to_tensor=True)
        semantic_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        semantic_score = max(0, min(100, semantic_sim * 100))

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

        # 3. Character-level similarity
        char_sim = Levenshtein.ratio(text1.lower(), text2.lower())
        char_score = char_sim * 100

        # 4. Length similarity
        len1, len2 = len(text1), len(text2)
        if len1 == 0 and len2 == 0:
            length_sim = 1.0
        else:
            length_sim = 1 - abs(len1 - len2) / max(len1, len2, 1)
        length_score = length_sim * 100

        # Overall score
        overall_score = (
            semantic_score * 0.5 +
            lexical_score * 0.25 +
            char_score * 0.15 +
            length_score * 0.10
        )

        return {
            'overall': round(overall_score, 1),
            'semantic': round(semantic_score, 1),
            'lexical': round(lexical_score, 1),
            'character': round(char_score, 1),
            'length': round(length_score, 1)
        }


class StyleConversionPractice:
    """Practice converting formal/standard sentences to your style."""

    def __init__(self, model_path: str, base_model: str = None):
        """Initialize practice session."""
        print("🤖 Loading your personalized AI model...")

        # Load metadata
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

        # Load vocabulary constraint (always use hard mode for this)
        self.vocab_constraint = get_constraint(self.tokenizer, model_path, 'hard')
        if self.vocab_constraint is None:
            print("⚠️  WARNING: No vocabulary constraint available!")
            print("Run: python scripts/extract_vocabulary.py")
            print("This practice mode works best with vocabulary constraints.\n")

        # Initialize similarity scorer
        self.scorer = SimilarityScorer()

        # Track scores
        self.scores = []

        # Formal prompts to convert
        self.prompts = self._get_conversion_prompts()

        print("✓ Practice mode ready!\n")

    def _get_conversion_prompts(self) -> list:
        """Get formal sentences to convert to casual style."""
        return [
            # Time and scheduling
            "I will be arriving home at approximately 6:00 PM this evening.",
            "What time should I expect you to return?",
            "I am currently running behind schedule.",
            "Please inform me when you have completed your tasks.",
            "I will be available later this afternoon.",

            # Food and dining
            "Have you had your dinner yet?",
            "What would you like to eat for lunch today?",
            "I am quite hungry at the moment.",
            "Shall we order some food for delivery?",
            "I will prepare something to eat shortly.",

            # Plans and activities
            "Do you have any plans for this weekend?",
            "I am going to the shopping mall tomorrow.",
            "Would you like to accompany me?",
            "I need to purchase some groceries.",
            "Let's meet up later if you are free.",

            # Status updates
            "I am currently occupied with something.",
            "I will contact you in a few minutes.",
            "I did not see your previous message.",
            "I am on my way right now.",
            "I will be there very soon.",

            # Questions and requests
            "Can you assist me with something?",
            "Do you know where I placed my keys?",
            "Have you completed your homework?",
            "Please remember to bring that item.",
            "Did you understand what I mentioned earlier?",

            # Responses and acknowledgments
            "Yes, I understand completely.",
            "That sounds acceptable to me.",
            "I am not entirely sure about that.",
            "Alright, that works for me.",
            "I will take care of it.",

            # Negative/problems
            "I cannot locate my phone anywhere.",
            "This is not functioning properly.",
            "I forgot to complete that task.",
            "I am experiencing some difficulties.",
            "That did not work out as planned.",

            # Casual conversation
            "How was your day today?",
            "That is quite amusing.",
            "I am feeling rather tired right now.",
            "The weather is quite pleasant today.",
            "I will speak with you later.",
        ]

    def convert_to_your_style(self, formal_text: str) -> str:
        """
        Use the model with vocab constraints to convert formal text to your style.
        """
        # Create prompt for style conversion
        prompt = f"""<s>[INST] Rewrite this sentence in a casual, natural way as if you're texting someone:

"{formal_text}"

Your casual version: [/INST] """

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(self.model.device)

        # Generate with vocabulary constraint
        generate_kwargs = {
            'max_new_tokens': 80,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            'repetition_penalty': 1.1
        }

        # Add vocabulary constraint if available
        if self.vocab_constraint is not None:
            generate_kwargs['logits_processor'] = [self.vocab_constraint]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract generated part
        response = response.split('[/INST]')[-1].strip()
        response = response.split('</s>')[0].strip()
        response = response.split('<s>')[0].strip()

        return response

    def run_practice_round(self, round_num: int):
        """Run a single practice round."""
        print("=" * 70)
        print(f"Round {round_num}")
        print("=" * 70)

        # Pick a random formal sentence
        formal_sentence = random.choice(self.prompts)

        print(f"\n📝 Formal sentence:")
        print(f'   "{formal_sentence}"')
        print("\n🎯 How would YOU say this casually?")
        print("(Type it exactly how you'd text it)\n")

        # Get user's conversion
        user_response = input("Your version: ").strip()

        if not user_response:
            print("\n⚠️  Empty response, skipping this round.\n")
            return None

        print("\n🤖 Generating AI's casual version (using YOUR vocabulary)...")

        # Generate model's conversion
        model_response = self.convert_to_your_style(formal_sentence)

        print(f"\n✨ AI's version (using only your words): \"{model_response}\"")

        # Calculate similarity
        print("\n📊 Calculating similarity...")
        scores = self.scorer.calculate_similarity(user_response, model_response)

        # Display results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n🎯 Overall Similarity: {scores['overall']}%")
        print("\nBreakdown:")
        print(f"  • Semantic (meaning):         {scores['semantic']}%")
        print(f"  • Lexical (word choice):      {scores['lexical']}%")
        print(f"  • Character (exact phrasing): {scores['character']}%")
        print(f"  • Length:                     {scores['length']}%")

        # Give feedback
        self._give_feedback(scores['overall'], user_response, model_response)

        print("=" * 70)

        return scores['overall']

    def _give_feedback(self, score: float, user_version: str, ai_version: str):
        """Give feedback based on score."""
        print("\n💭 Feedback:")
        if score >= 90:
            print("   🏆 Perfect match! You totally nailed your own style!")
        elif score >= 75:
            print("   🌟 Great! Very close to your natural style!")
        elif score >= 60:
            print("   👍 Good attempt! You're capturing your style.")
        elif score >= 40:
            print("   📚 Getting there! Keep practicing.")
        else:
            print("   🎯 Different approaches - compare the versions!")

        # Show what's different
        user_words = set(user_version.lower().split())
        ai_words = set(ai_version.lower().split())

        unique_to_user = user_words - ai_words
        unique_to_ai = ai_words - user_words

        if unique_to_user:
            print(f"\n   Words you used but AI didn't: {', '.join(list(unique_to_user)[:5])}")
        if unique_to_ai:
            print(f"   Words AI used but you didn't: {', '.join(list(unique_to_ai)[:5])}")

    def run_session(self, num_rounds: int = 5):
        """Run a full practice session."""
        print("\n" + "=" * 70)
        print("STYLE CONVERSION PRACTICE")
        print("=" * 70)
        print(f"\nYou'll see {num_rounds} formal sentences.")
        print("Convert each one to YOUR casual style!")
        print("Then see how the AI (using ONLY your words) would say it.")
        print("\nCommands:")
        print("  - Type your casual version")
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
            print("   You know your own style really well!")
            print("   The AI has learned to mimic you accurately!")
        elif avg_score >= 60:
            print("   Good grasp of your style!")
            print("   You and the AI are speaking similar casual language.")
        elif avg_score >= 40:
            print("   You're learning your patterns!")
            print("   Try to think about how you'd actually text it.")
        else:
            print("   Everyone has unique ways of speaking!")
            print("   Keep practicing to match your natural style.")

        print("\n👋 Thanks for practicing!")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Style Conversion Practice - Learn to match your own casual style'
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
        '--rounds',
        type=int,
        default=5,
        help='Number of practice rounds (default: 5)'
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"❌ Model not found at: {args.model_path}")
        print("Please train a model first using train.py")
        return

    # Check if vocabulary exists
    vocab_file = Path(args.model_path) / 'vocabulary.json'
    if not vocab_file.exists():
        print(f"⚠️  WARNING: Vocabulary file not found!")
        print("This mode works best with vocabulary constraints.")
        print("Run: python scripts/extract_vocabulary.py")
        print("\nContinuing anyway...\n")

    # Initialize practice session
    session = StyleConversionPractice(args.model_path, args.base_model)

    # Run practice
    session.run_session(args.rounds)


if __name__ == '__main__':
    main()
