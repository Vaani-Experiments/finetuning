import os
os.environ.setdefault('HF_HOME', './cache')

import re
import librosa
import evaluate

from datasets import load_dataset, concatenate_datasets
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

access_token = os.environ.get("HF_TOKEN")

modelTags = "openai/whisper-large-v3-turbo"
SAMPLE_RATE = 16000

processor = WhisperProcessor.from_pretrained(modelTags)
feature_extractor = WhisperFeatureExtractor.from_pretrained(modelTags)
tokenizer = processor.tokenizer

# Language map with fallbacks for unsupported languages
whisper_language_map = {
    "Angika":         {"iso_code": "anp", "whisper_supported": False, "fallback": "hi"},
    "Awadhi":         {"iso_code": "awa", "whisper_supported": False, "fallback": "hi"},
    "Bagri":          {"iso_code": "bgq", "whisper_supported": False, "fallback": "hi"},
    "Bajjika":        {"iso_code": "bjj", "whisper_supported": False, "fallback": "hi"},
    "Bearybashe":     {"iso_code": "byr", "whisper_supported": False, "fallback": "kn"},
    "Bengali":        {"iso_code": "bn",  "whisper_supported": True,  "fallback": "bn"},
    "Bhatri":         {"iso_code": "bgw", "whisper_supported": False, "fallback": "hi"},
    "Bhili":          {"iso_code": "bhb", "whisper_supported": False, "fallback": "hi"},
    "Bhojpuri":       {"iso_code": "bho", "whisper_supported": False, "fallback": "hi"},
    "Bundeli":        {"iso_code": "bns", "whisper_supported": False, "fallback": "hi"},
    "Chhattisgarhi":  {"iso_code": "hne", "whisper_supported": False, "fallback": "hi"},
    "Garhwali":       {"iso_code": "gbm", "whisper_supported": False, "fallback": "hi"},
    "Gondi":          {"iso_code": "gon", "whisper_supported": False, "fallback": "hi"},
    "Gujarati":       {"iso_code": "gu",  "whisper_supported": True,  "fallback": "gu"},
    "Halbi":          {"iso_code": "hlb", "whisper_supported": False, "fallback": "hi"},
    "Hindi":          {"iso_code": "hi",  "whisper_supported": True,  "fallback": "hi"},
    "Kannada":        {"iso_code": "kn",  "whisper_supported": True,  "fallback": "kn"},
    "Khariboli":      {"iso_code": "hi",  "whisper_supported": True,  "fallback": "hi"},
    "Khortha":        {"iso_code": "kht", "whisper_supported": False, "fallback": "hi"},
    "Konkani":        {"iso_code": "kok", "whisper_supported": False, "fallback": "hi"},
    "Kumaoni":        {"iso_code": "kfy", "whisper_supported": False, "fallback": "hi"},
    "Kurmali":        {"iso_code": "unr", "whisper_supported": False, "fallback": "hi"},
    "Kurukh":         {"iso_code": "kru", "whisper_supported": False, "fallback": "sat"},
    "Magahi":         {"iso_code": "mag", "whisper_supported": False, "fallback": "hi"},
    "Maithili":       {"iso_code": "mai", "whisper_supported": False, "fallback": "hi"},
    "Malvani":        {"iso_code": "mwr", "whisper_supported": False, "fallback": "mr"},
    "Marathi":        {"iso_code": "mr",  "whisper_supported": True,  "fallback": "mr"},
    "Marwari":        {"iso_code": "mwr", "whisper_supported": False, "fallback": "hi"},
    "Rajasthani":     {"iso_code": "raj", "whisper_supported": False, "fallback": "hi"},
    "Sadri":          {"iso_code": "sck", "whisper_supported": False, "fallback": "sat"},
    "Santali":        {"iso_code": "sat", "whisper_supported": True,  "fallback": "sat"},
    "Shekhawati":     {"iso_code": "swv", "whisper_supported": False, "fallback": "hi"},
    "Surgujia":       {"iso_code": "sgj", "whisper_supported": False, "fallback": "hi"},
    "Surjapuri":      {"iso_code": "sjp", "whisper_supported": False, "fallback": "hi"},
    "Tamil":          {"iso_code": "ta",  "whisper_supported": True,  "fallback": "ta"},
    "Telugu":         {"iso_code": "te",  "whisper_supported": True,  "fallback": "te"},
    "Tulu":           {"iso_code": "tcy", "whisper_supported": False, "fallback": "kn"},
    "Urdu":           {"iso_code": "ur",  "whisper_supported": True,  "fallback": "ur"},
    "Chakma":         {"iso_code": "tcy", "whisper_supported": False, "fallback": "en"},
}


def filter_long_sequences(example):
    tokenized_input = tokenizer(example["transcript"], truncation=False)
    return len(tokenized_input["input_ids"]) <= 440


def remove_text_between_delimiters(text):
    try:
        pattern = r'\(.*?\)|\[.*?\]|\{.*?\}|<.*?>'
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = re.sub(r"❴.*?❵", "", cleaned_text)
        characters_to_remove = r'[\:\,\.\!\-\"\?\)\(\}\(\!\t\n\।]'
        chars_to_remove_regex = r'[,?.!;:"“%‘”�\'-]'
        cleaned_text = re.sub(characters_to_remove, '', cleaned_text)
        cleaned_text = re.sub(chars_to_remove_regex, '', cleaned_text)
        cleaned_text = re.sub(' +', ' ', cleaned_text)
        return cleaned_text.strip()
    except Exception:
        return ''


def remove_special_characters(batch):
    batch["transcript"] = remove_text_between_delimiters(batch["transcript"])
    return batch


metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    filtered_preds, filtered_refs = [], []
    for p, r in zip(pred_str, label_str):
        if r.strip() != "":
            filtered_preds.append(p)
            filtered_refs.append(r)

    if len(filtered_refs) == 0:
        print("⚠️ All reference strings are empty — skipping WER computation.")
        return {"wer": 100.0}

    wer = 100 * metric.compute(predictions=filtered_preds, references=filtered_refs)
    return {"wer": wer}


def prepare_dataset(batch):
    audio = batch["audio"]
    sr = SAMPLE_RATE
    audio_array = librosa.resample(audio["array"], orig_sr=audio["sampling_rate"], target_sr=sr) if audio["sampling_rate"] != sr else audio["array"]
    batch["input_features"] = feature_extractor(audio_array, sampling_rate=sr).input_features[0]

    labels = processor.tokenizer(batch["transcript"]).input_ids
    batch["labels"] = labels
    batch["attention_mask"] = [1] * len(labels)

    language_name = batch.get("language", "Hindi")
    fallback_lang = whisper_language_map.get(language_name, {"fallback": "en"})["fallback"]
    print(fallback_lang)
    print(batch["transcript"])
    exit()
    batch["forced_decoder_ids"] = processor.get_decoder_prompt_ids(language=fallback_lang, task="transcribe")
    return batch


class MultilingualWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        forced_decoder_ids = inputs.pop("forced_decoder_ids", None)
        if forced_decoder_ids:
            model.config.forced_decoder_ids = forced_decoder_ids[0]
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v for k, v in inputs.items() if k != "forced_decoder_ids"}
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        for item in batch:
            if "forced_decoder_ids" not in item:
                item["forced_decoder_ids"] = self.processor.get_decoder_prompt_ids(
                    language="en", task="transcribe"
                )

        input_features = [b["input_features"] for b in batch]
        labels = [b["labels"] for b in batch]
        attention_masks = [b["attention_mask"] for b in batch]
        forced_decoder_ids = batch[0]["forced_decoder_ids"]

        batch_output = self.processor.feature_extractor.pad(
            {"input_features": input_features}, return_tensors="pt"
        )
        label_output = self.processor.tokenizer.pad(
            {"input_ids": labels, "attention_mask": attention_masks},
            padding=True,
            return_tensors="pt",
        )

        batch_output["labels"] = label_output["input_ids"]
        batch_output["attention_mask"] = label_output["attention_mask"]
        batch_output["forced_decoder_ids"] = forced_decoder_ids
        return batch_output


lanlist = ["Bhojpuri"]

dataset_train_list = []
dataset_test_list = []
for lan in lanlist:
    trainFolder = f"./vaaniDataset/{lan}/train"
    testFolder = f"./vaaniDataset/{lan}/validation"
    print(lan)

    dataset = load_dataset("audiofolder", data_dir=trainFolder)['train']
    dataset_train_subset = dataset.map(remove_special_characters).filter(filter_long_sequences).map(prepare_dataset, remove_columns=dataset.column_names)
    dataset = load_dataset("audiofolder", data_dir=testFolder)['train']
    dataset_test_subset = dataset.map(remove_special_characters).filter(filter_long_sequences).map(prepare_dataset, remove_columns=dataset.column_names)
    dataset_test_subset.save_to_disk(f"./processedDataset/{lan}/valid")
    dataset_train_subset.save_to_disk(f"./processedDataset/{lan}/train")

    dataset_train_list.append(dataset_train_subset)
    dataset_test_list.append(dataset_test_subset)

dataset_train = concatenate_datasets(dataset_train_list).shuffle(seed=42)
dataset_test = concatenate_datasets(dataset_test_list).shuffle(seed=42)

exit()

model = WhisperForConditionalGeneration.from_pretrained(modelTags)

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-v3-multilingual-vaani-turbo",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=200,
    num_train_epochs=20,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=128,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=4000,
    eval_steps=4000,
    logging_steps=400,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

data_collator = WhisperDataCollator(processor)

trainer = MultilingualWhisperTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
