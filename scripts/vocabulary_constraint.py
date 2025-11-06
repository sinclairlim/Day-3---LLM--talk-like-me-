"""
Vocabulary constraint for generation.
Ensures the model only generates tokens that correspond to words in your vocabulary.
"""

import json
import torch
from pathlib import Path
from transformers import LogitsProcessor
from typing import Set, List
import re


class VocabularyConstraint(LogitsProcessor):
    """
    Logits processor that constrains generation to use only words from vocabulary.

    This works by:
    1. Loading your vocabulary (words you've actually used)
    2. During generation, penalizing tokens that would create out-of-vocabulary words
    3. Allowing common punctuation and special tokens
    """

    def __init__(self, tokenizer, vocabulary_file: str, penalty: float = -100.0):
        """
        Initialize vocabulary constraint.

        Args:
            tokenizer: The model's tokenizer
            vocabulary_file: Path to vocabulary.json
            penalty: Penalty for out-of-vocabulary tokens (very negative = blocked)
        """
        self.tokenizer = tokenizer
        self.penalty = penalty

        # Load vocabulary
        with open(vocabulary_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.vocabulary = set(vocab_data['vocabulary'])

        print(f"🔒 Loaded vocabulary constraint: {len(self.vocabulary)} unique words")

        # Create allowed token IDs
        self.allowed_token_ids = self._create_allowed_token_ids()

        # Always allow special tokens
        self.special_token_ids = set([
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        ])
        # Remove None values
        self.special_token_ids = {tid for tid in self.special_token_ids if tid is not None}

    def _create_allowed_token_ids(self) -> Set[int]:
        """
        Create set of allowed token IDs based on vocabulary.

        This is tricky because tokens don't always map 1:1 to words.
        We need to be lenient to allow the model to generate fluently.
        """
        allowed = set()

        # Get all token IDs from vocabulary
        vocab_size = len(self.tokenizer)

        for token_id in range(vocab_size):
            token = self.tokenizer.decode([token_id]).strip().lower()

            # Remove special characters for matching
            token_clean = re.sub(r'[^\w\s]', '', token)

            # Allow if:
            # 1. Token is in vocabulary
            # 2. Token is a substring of a vocabulary word (for subword tokens)
            # 3. Token is punctuation or whitespace
            # 4. Token is very short (likely a subword)

            if not token or not token_clean:
                # Whitespace, punctuation, etc
                allowed.add(token_id)
                continue

            if token_clean in self.vocabulary:
                allowed.add(token_id)
                continue

            # Check if token is part of any vocabulary word
            if len(token_clean) <= 3:  # Short tokens are likely subwords
                allowed.add(token_id)
                continue

            # Check if this token is a substring of any vocab word
            is_subword = False
            for vocab_word in self.vocabulary:
                if token_clean in vocab_word or vocab_word in token_clean:
                    is_subword = True
                    break

            if is_subword:
                allowed.add(token_id)

        return allowed

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to penalize out-of-vocabulary tokens.

        Args:
            input_ids: Previously generated token IDs
            scores: Logits for next token

        Returns:
            Modified scores with penalties applied
        """
        # Create mask for disallowed tokens
        vocab_size = scores.shape[-1]
        mask = torch.ones(vocab_size, device=scores.device, dtype=torch.bool)

        # Allow vocabulary tokens and special tokens
        for token_id in self.allowed_token_ids:
            if token_id < vocab_size:
                mask[token_id] = False

        for token_id in self.special_token_ids:
            if token_id < vocab_size:
                mask[token_id] = False

        # Apply penalty to disallowed tokens
        scores = scores.masked_fill(mask, self.penalty)

        return scores


def load_vocabulary_constraint(tokenizer, model_path: str) -> VocabularyConstraint:
    """
    Load vocabulary constraint from model directory.

    Args:
        tokenizer: Model tokenizer
        model_path: Path to fine-tuned model directory

    Returns:
        VocabularyConstraint instance or None if vocabulary file not found
    """
    vocab_file = Path(model_path) / 'vocabulary.json'

    if not vocab_file.exists():
        print(f"⚠️  Vocabulary file not found at {vocab_file}")
        print("Run extract_vocabulary.py first to create it.")
        return None

    return VocabularyConstraint(tokenizer, str(vocab_file))


class SoftVocabularyConstraint(VocabularyConstraint):
    """
    Softer vocabulary constraint that penalizes but doesn't block unknown words.
    This allows more flexibility while still preferring known vocabulary.
    """

    def __init__(self, tokenizer, vocabulary_file: str, penalty: float = -5.0):
        """
        Initialize soft constraint.

        Args:
            tokenizer: The model's tokenizer
            vocabulary_file: Path to vocabulary.json
            penalty: Softer penalty for out-of-vocabulary tokens (default: -5.0)
        """
        # Use a softer penalty
        super().__init__(tokenizer, vocabulary_file, penalty=penalty)
        print("  Using SOFT constraint mode (allows unknown words with penalty)")


def get_constraint(tokenizer, model_path: str, mode: str = 'hard') -> VocabularyConstraint:
    """
    Get vocabulary constraint based on mode.

    Args:
        tokenizer: Model tokenizer
        model_path: Path to fine-tuned model
        mode: 'hard' (block unknown words), 'soft' (penalize), or 'none' (no constraint)

    Returns:
        Constraint instance or None
    """
    if mode == 'none':
        return None

    vocab_file = Path(model_path) / 'vocabulary.json'

    if not vocab_file.exists():
        print(f"⚠️  Vocabulary file not found at {vocab_file}")
        print("Run: python scripts/extract_vocabulary.py")
        return None

    if mode == 'soft':
        return SoftVocabularyConstraint(tokenizer, str(vocab_file))
    else:  # hard
        return VocabularyConstraint(tokenizer, str(vocab_file))
