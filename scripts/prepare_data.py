"""
Data preparation script for conversation-style fine-tuning.
Processes raw conversation data into a format suitable for LLM training.
"""

import json
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict
import argparse
from datetime import datetime


def parse_whatsapp_format(lines: List[str], your_name: str = "Sinclair") -> List[Dict[str, str]]:
    """
    Parse WhatsApp chat export format.
    Format: [DD/MM/YY, HH:MM:SS AM/PM] Name: Message

    Args:
        lines: Lines from the WhatsApp chat file
        your_name: Your name as it appears in the chat

    Returns:
        List of conversation turns
    """
    conversations = []

    # WhatsApp message pattern: [date, time] Name: Message
    # Pattern matches: [27/5/17, 4:17:49 AM] Mom: Message text
    whatsapp_pattern = r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)?)\]\s*([^:]+):\s*(.+)$'

    current_speaker = None
    current_message = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip system messages
        if '‎Messages and calls are end-to-end encrypted' in line:
            continue
        if line.startswith('‎'):  # Skip other WhatsApp system messages
            continue

        match = re.match(whatsapp_pattern, line)

        if match:
            # Save previous message
            if current_speaker and current_message:
                conversations.append({
                    'speaker': current_speaker,
                    'message': ' '.join(current_message)
                })

            # Extract new message parts
            date, time, name, message = match.groups()
            name = name.strip()

            # Determine speaker
            if name.lower() == your_name.lower():
                current_speaker = 'you'
            elif 'mom' in name.lower() or 'mother' in name.lower():
                current_speaker = 'mom'
            else:
                # Unknown speaker, try to infer
                current_speaker = 'mom'  # Default to mom for unknown speakers

            current_message = [message.strip()]
        else:
            # Continuation of previous message (multi-line message)
            if current_speaker and line:
                current_message.append(line)

    # Add last message
    if current_speaker and current_message:
        conversations.append({
            'speaker': current_speaker,
            'message': ' '.join(current_message)
        })

    return conversations


def parse_conversation_file(file_path: str, your_name: str = "Sinclair") -> List[Dict[str, str]]:
    """
    Parse conversation data from various formats.
    Expected format: Each conversation should have a clear distinction between
    your messages and your mom's messages.

    Supports:
    - JSON: [{"speaker": "you/mom", "message": "text"}, ...]
    - CSV: columns "speaker" and "message"
    - TXT: Plain text with markers like "You:" and "Mom:" OR WhatsApp export format

    Args:
        file_path: Path to the conversation file
        your_name: Your name as it appears in WhatsApp chats (for auto-detection)
    """
    file_path = Path(file_path)

    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        return df.to_dict('records')

    elif file_path.suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Detect if it's WhatsApp format by checking first few lines
        is_whatsapp = False
        for line in lines[:10]:
            if re.match(r'^\[\d{1,2}/\d{1,2}/\d{2,4},', line):
                is_whatsapp = True
                break

        if is_whatsapp:
            print("📱 Detected WhatsApp chat format")
            return parse_whatsapp_format(lines, your_name)

        # Otherwise, parse as simple format
        conversations = []
        current_speaker = None
        current_message = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect speaker changes (adjust patterns as needed)
            if line.lower().startswith('you:') or line.lower().startswith('me:'):
                if current_speaker and current_message:
                    conversations.append({
                        'speaker': current_speaker,
                        'message': ' '.join(current_message)
                    })
                current_speaker = 'you'
                current_message = [line.split(':', 1)[1].strip()]

            elif line.lower().startswith('mom:') or line.lower().startswith('mother:'):
                if current_speaker and current_message:
                    conversations.append({
                        'speaker': current_speaker,
                        'message': ' '.join(current_message)
                    })
                current_speaker = 'mom'
                current_message = [line.split(':', 1)[1].strip()]

            else:
                # Continue previous message
                if current_speaker:
                    current_message.append(line)

        # Add last message
        if current_speaker and current_message:
            conversations.append({
                'speaker': current_speaker,
                'message': ' '.join(current_message)
            })

        return conversations

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def create_training_examples(conversations: List[Dict[str, str]],
                            context_window: int = 3) -> List[Dict[str, str]]:
    """
    Convert conversations into training examples.
    Each example includes context (previous messages) and the target response.

    Args:
        conversations: List of conversation turns
        context_window: Number of previous messages to include as context

    Returns:
        List of training examples with 'prompt' and 'response' keys
    """
    training_examples = []

    for i in range(len(conversations)):
        if conversations[i]['speaker'] != 'you':
            continue

        # Get context (previous messages)
        context_start = max(0, i - context_window)
        context = conversations[context_start:i]

        # Build prompt from context
        prompt_parts = []
        for turn in context:
            speaker = "You" if turn['speaker'] == 'you' else "Mom"
            prompt_parts.append(f"{speaker}: {turn['message']}")

        # Add the current turn's prompt
        if i > 0:
            prompt_parts.append(f"Mom: {conversations[i-1]['message']}")

        prompt = "\n".join(prompt_parts)
        response = conversations[i]['message']

        training_examples.append({
            'prompt': prompt.strip(),
            'response': response.strip()
        })

    return training_examples


def format_for_instruction_tuning(examples: List[Dict[str, str]]) -> List[str]:
    """
    Format examples for instruction-tuning format.
    This uses a chat template that most models expect.
    """
    formatted = []

    for example in examples:
        # Format as instruction-following conversation
        text = f"""<s>[INST] You are responding in a conversation. Continue the conversation in your natural speaking style.

{example['prompt']}

Your response: [/INST] {example['response']}</s>"""

        formatted.append(text)

    return formatted


def save_training_data(examples: List[str], output_path: str,
                       train_split: float = 0.9):
    """
    Save formatted examples and split into train/validation sets.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    import random
    random.shuffle(examples)

    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    # Save as JSONL
    with open(output_path / 'train.jsonl', 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps({'text': example}) + '\n')

    with open(output_path / 'validation.jsonl', 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps({'text': example}) + '\n')

    print(f"✓ Saved {len(train_examples)} training examples")
    print(f"✓ Saved {len(val_examples)} validation examples")
    print(f"✓ Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare conversation data for LLM fine-tuning'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input conversation file (JSON, CSV, TXT, or WhatsApp export)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--your-name',
        type=str,
        default='Sinclair',
        help='Your name as it appears in the chat (for WhatsApp format)'
    )
    parser.add_argument(
        '--context-window',
        type=int,
        default=3,
        help='Number of previous messages to include as context'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.9,
        help='Proportion of data to use for training (rest for validation)'
    )

    args = parser.parse_args()

    print(f"📚 Loading conversations from {args.input}...")
    conversations = parse_conversation_file(args.input, args.your_name)
    print(f"✓ Loaded {len(conversations)} conversation turns")

    # Show some statistics
    you_count = sum(1 for c in conversations if c['speaker'] == 'you')
    mom_count = sum(1 for c in conversations if c['speaker'] == 'mom')
    print(f"  - Your messages: {you_count}")
    print(f"  - Mom's messages: {mom_count}")

    print(f"\n🔄 Creating training examples...")
    examples = create_training_examples(conversations, args.context_window)
    print(f"✓ Created {len(examples)} training examples")

    if len(examples) == 0:
        print("\n❌ No training examples created! Check your input format and --your-name parameter.")
        return

    print(f"\n📝 Formatting for instruction tuning...")
    formatted = format_for_instruction_tuning(examples)

    print(f"\n💾 Saving training data...")
    save_training_data(formatted, args.output, args.train_split)

    print("\n✅ Data preparation complete!")


if __name__ == '__main__':
    main()
