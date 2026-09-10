# ASSETS — big files live on Hugging Face, this manifest points at them
#
# Format:  <local path> \t <HF resolve URL>
# `uv run fetch_assets.py` downloads every entry not already present.
# `HF_TOKEN=... uv run publish_sft.py --from <lab dir>` uploads the two
# checkpoints + SFT corpus repo if they aren't up yet.
#
# Model repo:    https://huggingface.co/NathanielArfin/gpt-hansard-11m
# SFT dataset:   https://huggingface.co/datasets/NathanielArfin/gpt-hansard-11m-sft

gpt-11m-base.pt	https://huggingface.co/NathanielArfin/gpt-hansard-11m/resolve/main/gpt-11m-base.pt
gpt-11m-sft-en.pt	https://huggingface.co/NathanielArfin/gpt-hansard-11m/resolve/main/gpt-11m-sft-en.pt
gpt-11m-sft-bilingual.pt	https://huggingface.co/NathanielArfin/gpt-hansard-11m/resolve/main/gpt-11m-sft-bilingual.pt
gpt-11m-sft-bilingual-named.pt	https://huggingface.co/NathanielArfin/gpt-hansard-11m/resolve/main/gpt-11m-sft-bilingual-named.pt
gpt-11m-sft-en-v3.pt	https://huggingface.co/NathanielArfin/gpt-hansard-11m/resolve/main/gpt-11m-sft-en-v3.pt
sft-stream-en.bin	https://huggingface.co/datasets/NathanielArfin/gpt-hansard-11m-sft/resolve/main/sft-stream-en.bin
sft-stream-en.bin.labels	https://huggingface.co/datasets/NathanielArfin/gpt-hansard-11m-sft/resolve/main/sft-stream-en.bin.labels
sft/qp-pairs.jsonl	https://huggingface.co/datasets/NathanielArfin/gpt-hansard-11m-sft/resolve/main/qp-pairs.jsonl
sft/qp-heldout.jsonl	https://huggingface.co/datasets/NathanielArfin/gpt-hansard-11m-sft/resolve/main/qp-heldout.jsonl