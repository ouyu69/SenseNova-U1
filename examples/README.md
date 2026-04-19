# Examples

Reference inference scripts for SenseNova-U1. Every script here is intentionally
self-contained — on top of the `sensenova_u1` package itself it only pulls in
`torch`, `transformers`, `pillow`, `numpy` (and optionally `tqdm` /
`flash-attn`).

Each task lives in its own subfolder with a matching `data/` directory of
sample inputs:

```
examples/
├── README.md
├── t2i/                       # text-to-image               (runnable)
│   ├── inference.py
│   ├── run.sh
│   └── data/
│       └── samples.jsonl
├── ti2i/                      # image editing               (TBA)
│   ├── inference.py
│   └── data/
│       └── samples.jsonl
├── interleave/                # interleaved generation      (TBA)
│   ├── inference.py
│   └── data/
│       └── samples.jsonl
└── vqa/                       # visual understanding / VQA  (TBA)
    ├── inference.py
    └── data/
        └── questions.jsonl
```

## Text-to-Image

Single prompt:

```bash
python examples/t2i/inference.py \
  --model_path OpenSenseNova/SenseNova-U1-Mini \
  --prompt "一个咖啡店门口有一个黑板，上面写着日日新咖啡，2元一杯，旁边有个霓虹灯，写着商汤科技，旁边有个海报，海报上面是一只小浣熊，海报下方写着SenseNova newbee。" \
  --width 2048 --height 2048 \
  --output out.png \
  --profile
```

Batched prompts from a JSONL file (each line must contain a `prompt`;
`width` / `height` / `seed` are optional):

```bash
python examples/t2i/inference.py \
    --model_path OpenSenseNova/SenseNova-U1-Mini \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --profile
```

See [`t2i/data/samples.jsonl`](./t2i/data/samples.jsonl) for a tiny starter
file. Supported resolution buckets and the full CLI flag list live in the
top-level [README](../README.md#text-to-image).
