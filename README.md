# NanoChat

A macOS app that runs the nanochat d34 base model (2.2B parameters) on Apple Silicon.
It loads the original PyTorch checkpoint and keeps it as BF16 or quantizes it to Q8 or Q4.
No dependencies. Metal for the GPU, plain C for the rest.

## Files

| File | Function |
|---|---|
| `nc.h` | Shared types and the `.ncq` file format |
| `pt.c` | Reads the PyTorch checkpoint (zip + pickle) and `tokenizer.pkl`, quantizes, writes `.ncq` |
| `nc.c` | Maps a `.ncq` file, tokenizer (BPE), CPU forward pass, sampling |
| `gpu.m` | Metal forward pass, with the kernels (BF16/Q8/Q4 matvec, RMSNorm, RoPE + QK norm, attention, ReLU²) |
| `main.m` | The app (no NIB) |

## Procedure

1. Download the checkpoint. You need three files in one folder:

   ```
   mkdir -p ~/nanochat-d34 && cd ~/nanochat-d34
   curl -L -C - -O https://huggingface.co/karpathy/nanochat-d34/resolve/main/model_169150.pt
   curl -L -O https://huggingface.co/karpathy/nanochat-d34/resolve/main/meta_169150.json
   curl -L -O https://huggingface.co/karpathy/nanochat-d34/resolve/main/tokenizer.pkl
   ```

   If the download stops, run the first `curl` again. `-C -` continues it.

2. Open `NanoChat.xcodeproj` and run (Cmd-R).
3. Select **BF16**, **Q8** or **Q4**, then click **Import…** and select the folder.
   The app selects BF16 on a Mac with 16 GB or more, and Q8 on a Mac with 8 GB.
   The app writes a `.ncq` file and opens it. The import reads the checkpoint
   through `mmap`, one tensor at a time. Thus it is safe on 8 GB.
4. Type a prompt. Push **Cmd-Return** to generate. Push **Cmd-Return** or **Cmd-.** to stop.

The app opens the last model again at launch.

## Quantization

| Type | Size | Notes |
|---|---|---|
| BF16 | ~4.4 GB | No quantization. Same quality as the checkpoint. Use this on 16 GB. |
| Q8 | ~2.4 GB | int8, one bf16 scale per 32 values. Use this on 8 GB. Approximately 2× faster than BF16. |
| Q4 | ~1.3 GB | 4-bit, one bf16 scale per 32 values. `wte` and `lm_head` stay Q8. Lower quality. |

Speed is set by memory bandwidth: each token reads all the weights one time.
The KV cache for 2048 tokens is ~0.6 GB. On an 8 GB Mac, close other large apps before you run Q8.
After the import, you can delete `model_169150.pt`. The app uses only the `.ncq` file.

Q8 and Q4 files made with v1 open in v2 with no change.

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
- CPU forward: same logits as PyTorch with the same rounded weights, for BF16, Q8 and Q4 (max error 7e-4, fp16 KV cache).
- Error against the fp32 model (tiny test model, relative): BF16 0.4%, Q8 1.6%, Q4 20%.
- v2 writes Q8 and Q4 files that are identical to v1 files.
- Zip64 checkpoint larger than 4 GB: import gives an identical `.ncq`.
- `main.m` and `gpu.m` compile with no errors against the macOS 14.5 SDK.
- The Metal kernels are in `gpu.m` as a string. The app compiles them at launch
  (`newLibraryWithSource`), so the build does not need the Metal Toolchain.
  If a kernel does not compile, the app shows the compiler message and uses the CPU.
