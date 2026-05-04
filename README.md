# Vaani ASR Fine-tuning Experiments

Fine-tuning experiments on the [Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part)
multilingual Indian speech corpus, covering Whisper, FastConformer (NeMo) and Gemma-3n.

## Layout

| File | Stack | What it does |
| --- | --- | --- |
| `finetune_whisper_small.py` | HF Transformers | Fine-tunes `whisper-small` on a single Vaani language  |
| `finetune_whisper_large.py` | HF Transformers | Multilingual fine-tuning of `whisper-large-v3-turbo` with a language-fallback map for non-Whisper languages. |
| `finetune_gemma3n.py` | Unsloth + TRL | SFT of `gemma-3n-E2B-it` on multiple Vaani languages, formatted as audio chat-messages. |
| `finetune_fastconformer.py` | NVIDIA NeMo | Hydra entrypoint to fine-tune a FastConformer Hybrid TDT-CTC BPE model. |
| `fastconformer_hybrid_tdt_ctc_bpe.yaml` | NeMo config | Config consumed by the script above. |

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=<your huggingface token>     # needed to pull Vaani from HF
```

Each script expects local paths (datasets, tokenizer dirs, pretrained checkpoints) that are
specific to the original training environment — adjust the constants near the top of each
file before running.

## Running

```bash
# Whisper small / single language
python finetune_whisper_small.py

# Whisper large multilingual
python finetune_whisper_large.py

# Gemma-3n multilingual SFT
python finetune_gemma3n.py

# FastConformer (NeMo) — Hydra entrypoint
python finetune_fastconformer.py \
    init_from_nemo_model=path/to/pretrained.nemo \
    model.tokenizer.dir=path/to/tokenizer \
    model.train_ds.manifest_filepath=manifest_train.json \
    model.validation_ds.manifest_filepath=manifest_valid.json
```

## Notes

- Scripts read `HF_TOKEN` from the environment; never commit tokens.
- Caches, model checkpoints, datasets and manifests are ignored via `.gitignore`.
- The FastConformer config points at a tokenizer directory that must be built locally with
  NeMo's `process_asr_text_tokenizer.py`.
