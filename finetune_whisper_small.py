import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["HF_HOME"] = "./cache"

import warnings
warnings.filterwarnings("ignore")

import re
import librosa
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

print(f"CUDA Device Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

modelTags = "openai/whisper-small"
lan = "Hindi"

tokenizer = WhisperTokenizer.from_pretrained(modelTags, language=lan, task="transcribe")
tokenizer.model_max_length = 448
feature_extractor = WhisperFeatureExtractor.from_pretrained(modelTags)
processor = WhisperProcessor.from_pretrained(modelTags, language=lan, task="transcribe")

access_token = os.environ.get("HF_TOKEN")

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def remove_text_between_delimiters(text):
    try:
        pattern = r'\(.*?\)|\[.*?\]|\{.*?\}|<.*?>'
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = re.sub(r"❴.*?❵", "", cleaned_text)
        characters_to_remove = r'[\:\,\.\!\-\"\?\)\(\}\(\!\t\n\।]'
        chars_to_remove_regex = r'[,?.!;:"“%‘”�\'-]'
        cleaned_text = re.sub(characters_to_remove, '', cleaned_text)
        cleaned_text = re.sub(chars_to_remove_regex, '', cleaned_text)
        cleaned_text = re.sub(r'[A-Za-z0-9]', '', cleaned_text)
        cleaned_text = re.sub(' +', ' ', cleaned_text)
        return cleaned_text.strip()
    except Exception:
        return ''


def remove_special_characters(batch):
    batch["transcript"] = remove_text_between_delimiters(batch["transcript"])
    return batch


def filter_long_sequences(example):
    tokenized_input = tokenizer(example["transcript"], truncation=False)
    return len(tokenized_input["input_ids"]) <= 448


def prepare_dataset(batch):
    audio = batch["audio"]
    target_sr = 16000
    if audio["sampling_rate"] != target_sr:
        audio_array = librosa.resample(audio["array"], orig_sr=audio["sampling_rate"], target_sr=target_sr)
    else:
        audio_array = audio["array"]

    batch["input_features"] = feature_extractor(audio_array, sampling_rate=target_sr).input_features[0]
    labels = tokenizer(batch["transcript"]).input_ids
    batch["labels"] = labels
    batch["attention_mask"] = [1] * len(labels)
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        batch["attention_mask"] = labels_batch["attention_mask"]
        return batch


dataset_valid = load_dataset("audiofolder", data_dir="./dataset/coquiTTSout/valid")['train']
dataset_test = dataset_valid.filter(filter_long_sequences).map(prepare_dataset, remove_columns=dataset_valid.column_names)

dataset = load_dataset("audiofolder", data_dir="./dataset/coquiTTSout/train")['train']
dataset_train = dataset.filter(filter_long_sequences).map(prepare_dataset, remove_columns=dataset.column_names)

model = WhisperForConditionalGeneration.from_pretrained(modelTags)
model.generation_config.language = lan
model.generation_config.task = "transcribe"
model.config.use_cache = False

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-coquiTTS_alone",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=100,
    num_train_epochs=10,
    gradient_checkpointing=False,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    save_total_limit=2,
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()
