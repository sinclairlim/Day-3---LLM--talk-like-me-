"""
Extract vocabulary from training data to constrain model generation.
This ensures the model only uses words you've actually used before.
"""

import json
import re
from pathlib import Path
from collections import Counter
import argparse


def extract_vocabulary_from_training_data(data_dir: str, min_frequency: int = 1) -> dict:
    """
    Extract vocabulary from processed training data.

    Args:
        data_dir: Directory containing processed training data
        min_frequency: Minimum times a word must appear to be included

    Returns:
        dict with vocabulary stats and word list
    """
    train_file = Path(data_dir) / 'train.jsonl'

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at {train_file}")

    print(f"📚 Loading training data from {train_file}...")

    all_words = []
    all_responses = []

    # Read training data
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')

            # Extract your response (after [/INST])
            if '[/INST]' in text:
                response = text.split('[/INST]')[-1]
                response = response.replace('</s>', '').strip()

                if response:
                    all_responses.append(response)

                    # Tokenize into words
                    # Keep common contractions, abbreviations, and slang
                    words = re.findall(r"\b[\w']+\b|[^\w\s]", response.lower())
                    all_words.extend(words)

    print(f"✓ Processed {len(all_responses)} responses")

    # Count word frequencies
    word_freq = Counter(all_words)

    # Filter by minimum frequency
    vocabulary = {
        word: freq
        for word, freq in word_freq.items()
        if freq >= min_frequency
    }

    # Get unique words sorted by frequency
    sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)

    stats = {
        'total_responses': len(all_responses),
        'total_words': len(all_words),
        'unique_words': len(vocabulary),
        'min_frequency': min_frequency,
        'vocabulary': vocabulary,
        'top_words': sorted_vocab[:50],  # Top 50 most common words
    }

    return stats


def save_vocabulary(vocab_stats: dict, output_path: str):
    """Save vocabulary to file."""
    output_path = Path(output_path)

    # Save full vocabulary
    vocab_file = output_path / 'vocabulary.json'
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_responses': vocab_stats['total_responses'],
            'total_words': vocab_stats['total_words'],
            'unique_words': vocab_stats['unique_words'],
            'min_frequency': vocab_stats['min_frequency'],
            'vocabulary': list(vocab_stats['vocabulary'].keys()),
            'word_frequencies': vocab_stats['vocabulary']
        }, f, indent=2, ensure_ascii=False)

    # Save summary
    summary_file = output_path / 'vocabulary_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("VOCABULARY SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total responses analyzed: {vocab_stats['total_responses']}\n")
        f.write(f"Total words: {vocab_stats['total_words']}\n")
        f.write(f"Unique words (min freq {vocab_stats['min_frequency']}): {vocab_stats['unique_words']}\n\n")

        f.write("Top 50 Most Common Words:\n")
        f.write("-" * 70 + "\n")
        for i, (word, freq) in enumerate(vocab_stats['top_words'], 1):
            f.write(f"{i:3d}. {word:20s} (used {freq} times)\n")

    print(f"\n✓ Vocabulary saved to: {vocab_file}")
    print(f"✓ Summary saved to: {summary_file}")


def display_stats(vocab_stats: dict):
    """Display vocabulary statistics."""
    print("\n" + "=" * 70)
    print("VOCABULARY STATISTICS")
    print("=" * 70)
    print(f"\nTotal responses analyzed: {vocab_stats['total_responses']}")
    print(f"Total words: {vocab_stats['total_words']}")
    print(f"Unique words (min freq {vocab_stats['min_frequency']}): {vocab_stats['unique_words']}")

    print(f"\nTop 20 Most Common Words:")
    print("-" * 70)
    for i, (word, freq) in enumerate(vocab_stats['top_words'][:20], 1):
        print(f"{i:3d}. {word:20s} (used {freq} times)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Extract vocabulary from training data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed training data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/finetuned',
        help='Output directory for vocabulary file'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=1,
        help='Minimum times a word must appear to be included (default: 1)'
    )

    args = parser.parse_args()

    # Extract vocabulary
    vocab_stats = extract_vocabulary_from_training_data(
        args.data_dir,
        args.min_frequency
    )

    # Display stats
    display_stats(vocab_stats)

    # Save vocabulary
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    save_vocabulary(vocab_stats, output_path)

    print("\n✅ Vocabulary extraction complete!")
    print("\nYou can now use --use-vocabulary flag when chatting to constrain")
    print("the model to only use words from your training data.")


if __name__ == '__main__':
    main()
