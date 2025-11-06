# Talk Like Me - Personal LLM Fine-tuning

Train a language model to mimic your conversational style using conversation history.

## Overview

This project provides a complete pipeline for fine-tuning a small language model on your personal conversation data. It uses:

- **LoRA (Low-Rank Adaptation)** for efficient fine-tuning
- **4-bit quantization** for reduced memory usage
- **TinyLlama** as the default base model (can be changed)

## Project Structure

```
.
├── data/
│   ├── raw/              # Put your conversation data here
│   └── processed/        # Processed training data (auto-generated)
├── models/
│   └── finetuned/        # Saved fine-tuned models (auto-generated)
├── scripts/
│   ├── prepare_data.py        # Data preprocessing
│   ├── train.py               # Model training
│   ├── extract_vocabulary.py  # Extract your unique vocabulary
│   ├── vocabulary_constraint.py # Vocabulary constraint logic
│   ├── style_practice.py      # Style conversion practice (formal→casual)
│   ├── practice.py            # Conversation response practice
│   └── chat.py                # Interactive chat interface
└── requirements.txt
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you have an NVIDIA GPU, install PyTorch with CUDA support first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Prepare Your Conversation Data

Place your conversation data in the `data/raw/` directory. The script supports four formats:

#### Option A: WhatsApp Chat Export (Recommended!)
Export your WhatsApp chat and save as a `.txt` file. The script will automatically detect and parse it.

**How to export WhatsApp chat:**
1. Open your chat with your mom in WhatsApp
2. Tap the menu (three dots) → More → Export chat
3. Choose "Without Media"
4. Save the file to `data/raw/`

Example format:
```
[27/5/17, 4:17:49 AM] Mom: Pl update my mobile n Whatapp change to xxxxxxxx
[28/5/17, 10:49:42 PM] Sinclair: What you mean
[28/5/17, 10:50:27 PM] Mom: Stone hot windscreen while driving n got 2 small chips
[29/5/17, 12:02:14 AM] Sinclair: Huh then again must change?
```

#### Option B: JSON Format
```json
[
  {"speaker": "you", "message": "Hey mom, how was your day?"},
  {"speaker": "mom", "message": "It was great! I went to the market."},
  {"speaker": "you", "message": "oh nice! did you get those tomatoes you wanted?"}
]
```

#### Option C: CSV Format
```csv
speaker,message
you,Hey mom how was your day?
mom,It was great! I went to the market.
you,oh nice! did you get those tomatoes you wanted?
```

#### Option D: Plain Text Format
```
Mom: Hey, did you finish your homework?
You: yeah i finished it like an hour ago lol
Mom: Good! What do you want for dinner?
You: idk maybe pizza? or we could order thai
```

**Tips for good training data:**
- More data = better results (aim for 500+ conversation turns)
- Include a variety of topics and contexts
- Keep your natural writing style (typos, abbreviations, slang, etc.)
- WhatsApp exports work perfectly - no need to clean them up!
- Remove any sensitive/private information if needed

## Usage

### Step 1: Preprocess Your Data

For WhatsApp export:
```bash
python scripts/prepare_data.py \
    --input data/raw/whatsapp_chat.txt \
    --your-name "Sinclair" \
    --context-window 3
```

For other formats:
```bash
python scripts/prepare_data.py \
    --input data/raw/conversations.txt \
    --output data/processed \
    --context-window 3
```

**Arguments:**
- `--input`: Path to your conversation file (WhatsApp export, JSON, CSV, or TXT)
- `--your-name`: Your name as it appears in WhatsApp (default: "Sinclair")
- `--output`: Where to save processed data (default: `data/processed`)
- `--context-window`: Number of previous messages to include as context (default: 3)
- `--train-split`: Train/validation split ratio (default: 0.9)

The script will automatically detect WhatsApp format and parse it correctly!

### Step 2: Train the Model

```bash
python scripts/train.py \
    --data-dir data/processed \
    --output-dir models/finetuned \
    --epochs 3 \
    --batch-size 4
```

**Arguments:**
- `--data-dir`: Directory with processed training data
- `--model-name`: Base model to fine-tune (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- `--output-dir`: Where to save the fine-tuned model
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Batch size per device (default: 4)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--max-length`: Maximum sequence length (default: 512)
- `--no-4bit`: Disable 4-bit quantization (requires more GPU memory)

**Training time:** Depends on your data size and hardware:
- ~1000 examples on GPU: 10-30 minutes
- ~1000 examples on CPU: 2-4 hours

### Step 3: Extract Vocabulary (Optional but Recommended!)

Create a vocabulary file to constrain the model to only use words you've actually used:

```bash
python scripts/extract_vocabulary.py --data-dir data/processed --output models/finetuned
```

This analyzes your training data and creates a `vocabulary.json` file containing all words you've used. You can then use vocabulary constraints when chatting or practicing:

- **none**: No constraint (default) - model can use any words
- **soft**: Prefers your words but allows others if needed
- **hard**: Only uses words from your vocabulary - maximum authenticity!

### Step 4: Practice Modes (Fun!)

Choose between two practice modes to test your style:

#### Option A: Style Conversion Practice (Recommended!)

Convert formal sentences to your casual style and compare with the AI:

```bash
python scripts/style_practice.py --rounds 10
```

**How it works:**
1. You see a **formal sentence** like: *"I will be arriving home at approximately 6:00 PM"*
2. You convert it to **your casual style**: *"gonna be home like 6ish"*
3. The AI (using ONLY your vocabulary) shows its version: *"coming home round 6 lol"*
4. Get scored on similarity - higher = you know your own style well!

**Why this is cool:**
- Teaches you to recognize your own patterns
- AI uses vocabulary constraints (only YOUR words)
- More focused on style conversion than conversation
- Great for seeing how you naturally casualize language

**Arguments:**
- `--model-path`: Path to your fine-tuned model
- `--rounds`: Number of practice rounds (default: 5)

#### Option B: Conversation Response Practice

Test how the model learned your conversational responses:

```bash
python scripts/practice.py --model-path models/finetuned --rounds 5
```

**How it works:**
1. You see a message from "Mom" (from your real chat history)
2. Type what you would normally respond
3. The AI generates what it thinks you'd say
4. Get a similarity score (0-100%)

**Arguments:**
- `--model-path`: Path to your fine-tuned model
- `--data-dir`: Path to processed data (default: `data/processed`)
- `--rounds`: Number of practice rounds (default: 5)
- `--vocab-mode`: Vocabulary constraint mode: `none`, `soft`, or `hard`

**Similarity breakdown (both modes):**
- **Semantic (50%)**: Meaning and conversational style
- **Lexical (25%)**: Word choice and vocabulary
- **Character (15%)**: Exact phrasing and typos
- **Length (10%)**: Response length matching

### Step 5: Chat with Your Model

Have a free-form conversation with your AI:

```bash
python scripts/chat.py --model-path models/finetuned
```

**Arguments:**
- `--model-path`: Path to your fine-tuned model
- `--base-model`: Base model name (optional, auto-detected from metadata)
- `--temperature`: Response randomness, 0.0-2.0 (default: 0.7)
- `--max-tokens`: Max response length (default: 150)
- `--vocab-mode`: Vocabulary constraint mode: `none`, `soft`, or `hard`

**Chat commands:**
- Type your message and press Enter
- `quit` or `exit` - End the conversation
- `clear` - Clear conversation history
- `history` - View conversation history

**Using vocabulary constraints in chat:**
```bash
# Only use words you've actually used (maximum authenticity)
python scripts/chat.py --vocab-mode hard

# Prefer your words but allow others if needed
python scripts/chat.py --vocab-mode soft
```

## Hardware Requirements

### Minimum (CPU only):
- 8GB RAM
- Training will be slow

### Recommended:
- NVIDIA GPU with 8GB+ VRAM
- 16GB+ RAM
- CUDA support

### For larger models:
- NVIDIA GPU with 16GB+ VRAM
- Consider using models like `meta-llama/Llama-2-7b-chat-hf`

## Vocabulary Constraints - Maximum Authenticity

One unique feature is **vocabulary constraints** - ensuring the AI only uses words you've actually used!

### Why Use Vocabulary Constraints?

Without constraints, the model might use:
- Words you'd never say ("indeed", "furthermore", "subsequently")
- Formal language that doesn't match your casual style
- Synonyms you don't use

With vocabulary constraints, the AI is forced to:
- Only use words from your actual conversations
- Match your exact level of formality/casualness
- Use your specific slang, abbreviations, and typos

### Three Modes:

1. **None (default)**: No constraints - model can use any words
   - Pros: More fluent, handles any topic
   - Cons: Might not sound exactly like you

2. **Soft**: Prefers your vocabulary but allows other words if needed
   - Pros: Good balance of authenticity and fluency
   - Cons: Still might use some unfamiliar words

3. **Hard**: ONLY uses words from your vocabulary
   - Pros: Maximum authenticity - sounds most like you
   - Cons: Might struggle with topics you haven't discussed before

### Example:

Without constraint:
> "I'll be arriving home around 6 PM this evening"

With hard constraint (using your actual words):
> "gonna be home like 6ish lol"

## Customization

### Using a Different Base Model

You can use any causal language model from HuggingFace:

```bash
python scripts/train.py \
    --model-name "microsoft/phi-2" \
    --data-dir data/processed
```

**Recommended models:**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B params, fast, low memory)
- `microsoft/phi-2` (2.7B params, good quality)
- `meta-llama/Llama-2-7b-chat-hf` (7B params, best quality, requires more resources)

### Adjusting Training Parameters

For better quality (slower training):
```bash
python scripts/train.py \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-4
```

For faster training (may reduce quality):
```bash
python scripts/train.py \
    --epochs 2 \
    --batch-size 8 \
    --learning-rate 3e-4
```

## Tips for Best Results

1. **Data Quality**: Clean, consistent conversation data works best
2. **Data Quantity**: More examples = better results (aim for 1000+ turns)
3. **Training Duration**: Don't overtrain - monitor validation loss
4. **Temperature**: Lower (0.5-0.7) for consistent style, higher (0.8-1.0) for variety
5. **Context Window**: Adjust based on typical conversation length

## Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `--batch-size` to 2 or 1
- Reduce `--max-length` to 256
- Enable 4-bit quantization (default)
- Use a smaller base model

### Model Not Learning

- Increase `--epochs` (try 5-10)
- Ensure you have enough training data (500+ examples)
- Check data formatting is correct
- Try reducing `--learning-rate` to 1e-4

### Responses Don't Sound Like You

- Add more diverse training examples
- Increase training epochs
- Adjust `--context-window` during preprocessing
- Check that you're filtering for only YOUR messages in the data

### Training is Too Slow

- Use a GPU if available
- Reduce `--max-length`
- Increase `--batch-size` (if memory allows)
- Use a smaller base model

## Example Workflow

### Using WhatsApp Export:

```bash
# 1. Export your WhatsApp chat with your mom and save to data/raw/

# 2. Prepare data (replace "YourName" with your actual WhatsApp name)
python scripts/prepare_data.py \
    --input data/raw/whatsapp_chat.txt \
    --your-name "Sinclair" \
    --context-window 4

# 3. Train model
python scripts/train.py \
    --epochs 5 \
    --batch-size 4

# 4. Extract vocabulary (for authentic word usage)
python scripts/extract_vocabulary.py

# 5. Style conversion practice - Learn your patterns!
python scripts/style_practice.py --rounds 10

# 6. Or try conversation practice
python scripts/practice.py --rounds 10 --vocab-mode hard

# 7. Chat freely with your AI (using only your words!)
python scripts/chat.py --vocab-mode hard
```

### Using Other Formats:

```bash
# 1. Prepare data
python scripts/prepare_data.py \
    --input data/raw/conversations.txt \
    --context-window 4

# 2. Train model
python scripts/train.py \
    --epochs 5 \
    --batch-size 4

# 3. Extract vocabulary
python scripts/extract_vocabulary.py

# 4. Style conversion practice
python scripts/style_practice.py --rounds 10

# 5. Conversation practice
python scripts/practice.py --rounds 10 --vocab-mode hard

# 6. Chat mode
python scripts/chat.py --vocab-mode hard
```

## Privacy & Ethics

- Keep your training data private and secure
- Don't share models trained on private conversations
- Be mindful of what data you use for training
- This is for personal/educational use only

## License

MIT License - Feel free to modify and use for personal projects

## Acknowledgments

Built with:
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
