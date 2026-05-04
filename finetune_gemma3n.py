import os
import re
from typing import List, Dict

# HF cache directory (relative to repo). Change if you prefer an absolute path.
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "allhfCache")

import torch
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.cache_size_limit = 512

from unsloth import FastModel
from datasets import load_dataset, Audio, concatenate_datasets
from trl import SFTTrainer, SFTConfig


# -- Configuration -----------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/gemma-3n-E2B-it")
OUTPUT_DIR = "outputs_lr_5e-7"
BATCH_SIZE = 4
GRAD_ACCUM = 2
LR = 5e-7
NUM_EPOCHS = 1
SAMPLE_RATE = 16000
NUM_PROC_MAP = 4
VAANI_BASE = os.environ.get("VAANI_BASE", "./vaaniDataset")
VAANI_LANGS = ["Hindi", "Bengali", "Bhojpuri", "Chakma", "Kannada", "Telugu"]
# ---------------------------------------------------------------------------


# ---------------------- Text cleaning helpers -------------------------------
def remove_text_between_delimiters(text: str) -> str:
    """Remove bracketed text, punctuation, ASCII letters/digits and collapse spaces."""
    if not isinstance(text, str):
        return ""
    # remove content inside (), [], {}, <> and special pair ❴ ❵
    text = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}|<.*?>|❴.*?❵", "", text)
    # remove common punctuation and Devanagari danda (।) and whitespace control chars
    text = re.sub(r'[,?.!;:"“%‘”�\'\-\:\!\t\n\।]', "", text)
    # remove ASCII letters and digits
    text = re.sub(r"[A-Za-z0-9]", "", text)
    # normalize spaces
    return re.sub(r"\s+", " ", text).strip()


def remove_special_characters(batch: Dict) -> Dict:
    """Dataset map function: clean transcript field in-place."""
    if "transcript" in batch:
        batch["transcript"] = remove_text_between_delimiters(batch["transcript"])
    return batch
# ---------------------------------------------------------------------------


# --------------------- Dataset preparation ----------------------------------
def load_vaani_datasets(base_dir: str, langs: List[str], split_name: str = "train"):
    """Load and concatenate datasets from vaani-style layout: <base_dir>/<Lang>/<split>."""
    datasets = []
    for lang in langs:
        path = os.path.join(base_dir, lang, split_name)
        if not os.path.exists(path):
            # try load_dataset with dataset identifier if it's already a HF dataset path
            try:
                ds = load_dataset(path)
            except Exception:
                continue
        else:
            ds = load_dataset("audiofolder", data_dir=path)["train"]
        ds = ds.map(remove_special_characters, num_proc=NUM_PROC_MAP)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        datasets.append(ds)
    if not datasets:
        raise RuntimeError("No datasets found for the provided base_dir / langs.")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets).shuffle(seed=42)


def format_intersection_data(samples: Dict) -> Dict[str, List]:
    """Format audio+transcript dataset into the chat-message style expected by processor."""
    formatted = {"messages": []}
    audios = samples.get("audio", [])
    transcripts = samples.get("transcript", [])
    for i in range(len(audios)):
        audio = audios[i]["array"]
        label = str(transcripts[i])
        message = [
            {"role": "system", "content": [{"type": "text", "text": "You are an assistant that transcribes speech accurately."}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": "Please transcribe this audio."}]},
            {"role": "assistant", "content": [{"type": "text", "text": label}]},
        ]
        formatted["messages"].append(message)
    return formatted


def collate_fn(examples):
    """Data collator converting messages+audio into model inputs and labels."""
    texts, audios = [], []
    for ex in examples:
        text = processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False).strip()
        texts.append(text)
        audios.append(ex["audio"]["array"])
    batch = processor(text=texts, audio=audios, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    # mask padding and special tokens for loss
    ids_to_mask = [
        getattr(processor.tokenizer, "pad_token_id", None),
        getattr(processor.tokenizer, "image_token_id", None),
        getattr(processor.tokenizer, "audio_token_id", None),
        getattr(processor.tokenizer, "boi_token_id", None),
        getattr(processor.tokenizer, "eoi_token_id", None),
    ]
    for tid in set(x for x in ids_to_mask if x is not None):
        labels[labels == tid] = -100
    batch["labels"] = labels
    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(model.dtype)
    return batch
# ---------------------------------------------------------------------------


# ------------------------ Model & Trainer ----------------------------------
def initialize_model(model_path: str):
    """Load model + processor (FastModel)."""
    m, p = FastModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        max_seq_length=1024,
        load_in_4bit=True,
        full_finetuning=True,
    )
    return m, p


def make_trainer(model, dataset):
    cfg = SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.1,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=2,
        max_length=2048,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        args=cfg,
    )
    return trainer
# ---------------------------------------------------------------------------


def main():
    global model, processor

    # Initialize model & processor
    model, processor = initialize_model(MODEL_PATH)

    # Load and prepare datasets
    dataset = load_vaani_datasets(VAANI_BASE, VAANI_LANGS, split_name="train")
    dataset = dataset.map(format_intersection_data, batched=True, batch_size=4, num_proc=4)

    # Trainer and training
    trainer = make_trainer(model, dataset)
    trainer.train()

    # Save artifacts
    model.save_pretrained(os.path.join(OUTPUT_DIR, "model"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "processor"))
    print("Training finished. Artifacts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()