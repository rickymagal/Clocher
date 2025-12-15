from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/gpt-oss-20b",
    local_dir="models/gpt-oss-20b/hf",
    max_workers=1,
)
