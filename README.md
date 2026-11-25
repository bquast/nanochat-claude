# NanoChat

A macOS app that runs the nanochat d34 base model (2.2B parameters) on Apple Silicon.
No dependencies. Metal for the GPU, plain C for the rest.

## Files

| File | Function |
|---|---|
| `nc.h` | Shared types and the `.ncq` file format |
| `pt.c` | Reads the PyTorch checkpoint (zip + pickle) and `tokenizer.pkl`, quantizes, writes `.ncq` |
| `nc.c` | Maps a `.ncq` file, tokenizer (BPE), CPU forward pass, sampling |
| `gpu.m` | Metal forward pass |
| `kernels.metal` | Metal kernels: Q8/Q4 matvec, RMSNorm, RoPE + QK norm, attention, ReLU² |
| `main.m` | The app (no NIB) |

## Procedure

1. Download the checkpoint:

   ```
   pip install -U huggingface_hub
   hf download karpathy/nanochat-d34 --local-dir ~/nanochat-d34
   ```

   You need `model_169150.pt`, `meta_169150.json` and `tokenizer.pkl` in one folder.

2. Open `NanoChat.xcodeproj` and run (Cmd-R).
3. Select **Q8** or **Q4**, then click **Import…** and select the folder.
   The app writes a `.ncq` file and opens it. The import reads the checkpoint
   through `mmap`, one tensor at a time. Thus it is safe on 8 GB.
4. Type a prompt. Push **Cmd-Return** to generate. Push **Cmd-Return** or **Cmd-.** to stop.

The app opens the last model again at launch.

## Quantization

| Type | Size | Notes |
|---|---|---|
| Q8 | ~2.4 GB | int8, one bf16 scale per 32 values. Use this on 8 GB. |
| Q4 | ~1.3 GB | 4-bit, one bf16 scale per 32 values. `wte` and `lm_head` stay Q8. |

Close other large apps before you run Q8 on an 8 GB Mac. The KV cache for 2048 tokens is ~0.6 GB.

## Model

The code follows `gpt.py` at nanochat commit `2c4473d` (the d34 commit). The current
nanochat `master` has a different architecture. Do not use it as a reference for d34.

- `x = norm(wte[tok])`
- per layer: `x += Wo · attn(...)`, `x += Wpr · relu(Wfc · norm(x))²`
- RoPE base 10000 on (i, i + d/2) pairs, nanochat sign convention, then QK norm
- RMSNorm without weights, eps = float32 eps, no biases
- `logits = 15 · tanh(lm_head · norm(x) / 15)`

## Base model

This is a base model. It continues text. It does not answer like a chat model.
Write the start of a text and let it continue. The generated text is blue.
If you edit the page and generate again, the app runs only the changed part of the
prompt again (KV cache reuse).

## Engine

**Metal** is the default. **CPU** is the reference path (GCD, plain C). Use it to
compare output if you change a kernel. The CPU path is slower.

## Tests (done on Linux, not on a Mac)

- Tokenizer: same token ids as tiktoken on ~19k tokens of mixed text (code, CJK, accents, numbers, whitespace).
- CPU forward: same logits as PyTorch with the same quantized weights (max error 7e-4, fp16 KV cache).
- Zip64 checkpoint larger than 4 GB: import gives an identical `.ncq`.
- `main.m` and `gpu.m` compile with no errors against the macOS 14.5 SDK.
- `kernels.metal` is not compiled yet. Xcode compiles it on the first build.
