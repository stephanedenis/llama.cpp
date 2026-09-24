# Local tuning notes: 2x Quadro RTX 5000 (Turing, SM 7.5) + 2x Xeon E5-2687W v3

Machine-specific notes for this deployment. They are measurements, not upstream
guidance: every number below was taken on this host, and several of them
contradict what a reasonable default would suggest. Keep them with the machine.

Hardware: 2x Quadro RTX 5000 (16 GB each, compute capability 7.5, NVLink NV1
bonded at 25.8 GB/s), 2x Xeon E5-2687W v3 (10 cores each, 2 NUMA nodes),
125 GB RAM, CUDA 13.0, GCC 15, glibc 2.43.

## Host thread count: never combine many threads with `--numa distribute`

Measured with Qwen2.5-14B Q4_K_M, `-p 512 -n 128 -ngl 99 -fa on -r 3`, two GPUs,
layer split:

| Setting | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|
| `-t 40 --numa distribute` | **162.9** | 31.6 |
| `-t 40` (no `--numa`) | 1083.1 | 38.6 |
| `-t 10 --numa distribute` | 1075.6 | 37.9 |
| `-t 10` | 1090.4 | 38.6 |

Neither flag is wrong on its own; the *combination* collapses prompt processing
by a factor of 6.7. This was live in production and is the single largest
regression found on this host. Use `-t 20` with no `--numa` when the weights are
on the GPUs: the CPU only samples and prepares batches.

## Split mode: `tensor` is worth it on the NVLink pair

Same model and settings, `-t 10`:

| Setting | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|
| `-sm layer` | 1090.4 | 38.6 |
| `-sm tensor` | 1361.5 | 58.0 |
| `-sm tensor`, after NCCL was found | **1604.0** | **59.7** |
| `-sm layer`, after NCCL | 1094.2 | 39.2 |

`GGML_CUDA_P2P=1` made no measurable difference in either mode (1353 vs 1361 in
tensor, 1073 vs 1090 in layer), so it is not needed here.

Caveats: `--split-mode tensor` is marked experimental upstream, it disables
`--fit`, and `llama-cli -sm tensor` aborts in
`ggml-backend-meta.cpp:1799 (GGML_ASSERT(meta_buf_ctx->bufs[i]))` while
`llama-server` runs fine. Validate output quality before relying on it; a
temperature-0 comparison of the same prompt produced byte-identical output in
both modes on this host.

## NCCL

NCCL was never found by the original build (`NCCL_LIBRARY-NOTFOUND`), which
cost about 18 % of prompt processing in tensor mode. It is now installed
without root, from the wheel, into a plain directory:

```bash
python3 -m pip download nvidia-nccl-cu13 --no-deps -d /tmp/nccl-wheel
mkdir -p ~/.local/nccl && cd ~/.local/nccl
unzip -q /tmp/nccl-wheel/nvidia_nccl_cu13-*.whl
ln -sf libnccl.so.2 nvidia/nccl/lib/libnccl.so
```

and the build points at it, with the runpath embedded so no environment variable
is needed at runtime:

```bash
NCCL_DIR=$HOME/.local/nccl/nvidia/nccl
cmake -B build-cuda \
  -DNCCL_INCLUDE_DIR=$NCCL_DIR/include \
  -DNCCL_LIBRARY=$NCCL_DIR/lib/libnccl.so \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,$NCCL_DIR/lib" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$NCCL_DIR/lib"
```

Verify with `ldd build-cuda/bin/libggml-cuda.so.0.17.0 | grep nccl`.

## Storage

Model files live on `~/FastNVMe/models`. Measured raw sequential reads:
236 MB/s on the spinning `/home/stephane/Data13TB` volume against 1.8 GB/s on
the NVMe. Loading a 20 GB model costs about 85 s from the HDD and 11 s from the
NVMe, and `llama-bench` reloads the model for every test configuration.

## Fuzzy areas on this hardware

- `test-kvarn` fails with "did not exercise a head-wide store route". This is
  expected on Turing, not a regression: the route needs
  `smpbo >= KVAR_N_SHARED_BYTES`, which is 69704 bytes, while `sharedMemPerBlockOptin`
  on this GPU is 65536. The runtime check is correct and simply selects the
  low-shared-memory paths; the test's coverage assertion has no way to learn the
  device limit.
- FlashAttention always takes the vector kernels here, because
  `ggml_cuda_highest_compiled_arch()` is Turing when the build targets SM 7.5
  only. `-fa on` is still worth about 8 % prompt and 4 % generation over `-fa off`.
- `test-chat` fails in `test_template_output_peg_parsers`, before and after the
  changes recorded here. It is unrelated to the CUDA path.

## KVarN quality on the models actually hosted here

The fork documents KVarN against Qwen 3.6 and Gemma 4. The models on this host
are Qwen 2.5 and Mistral derivatives, so they were measured rather than assumed.
All of them have `head_dim 128` and a causal, non-MLA layout, which is what the
runtime check in `llama_init_from_model` asks for, and none of them needed a
special case.

Protocol: `llama-perplexity` on a 200 KB slice of `wiki.test.raw`, context 4096,
`-b 512 -ub 256`, four chunks (two chunks for the 32B model, which does not fit
in VRAM alongside the production servers), baseline `f16`/`f16` saved with
`--save-all-logits`, candidates replayed with `--kl-divergence` against that
same file. Every run used the identical corpus, batch sizes and cache family.

| Model | PPL f16 | Cache | Mean KLD | RMS Δp | Same top p |
|---|---:|---|---:|---:|---:|
| Codestral-22B | 5.2953 | kvarn4 | **0.001350** | 1.114 % | 98.41 % |
| | | kvarn4 + tail 1024 | **0.001023** | 0.999 % | 98.62 % |
| | | kvarn3 | 0.003439 | 1.762 % | 97.66 % |
| | | kvarn3 + tail 1024 | 0.001611 | 1.283 % | 98.21 % |
| | | kvarn2 | 0.016730 | 3.932 % | 94.81 % |
| | | kvarn2 + tail 1024 | 0.007574 | 2.821 % | 96.63 % |
| Qwen2.5-Coder-32B | 6.8171 | kvarn4 | 0.008304 | 2.699 % | 96.34 % |
| | | kvarn4 + tail 1024 | 0.001203 | 1.082 % | 98.61 % |
| | | kvarn3 | 0.011541 | 2.945 % | 95.87 % |
| | | kvarn3 + tail 1024 | 0.001713 | 1.209 % | 98.41 % |
| DeepSeek-R1-Distill-14B | 6.1787 | kvarn4 | 0.004297 | 1.947 % | 97.08 % |
| | | kvarn4 + tail 1024 | 0.003533 | 1.783 % | 97.52 % |
| | | kvarn3 | 0.008306 | 2.735 % | 96.34 % |
| | | kvarn3 + tail 1024 | 0.005086 | 2.161 % | 96.98 % |
| | | kvarn2 | 0.048065 | 6.503 % | 90.89 % |
| Qwen2.5-VL-7B | 8.0991 | kvarn4 | 0.005571 | 1.937 % | 96.96 % |
| | | kvarn4 + tail 1024 | 0.003981 | 1.730 % | 96.86 % |
| | | kvarn3 | 0.011422 | 2.972 % | 95.29 % |
| | | kvarn3 + tail 1024 | 0.005645 | 2.213 % | 96.67 % |
| | | kvarn2 | 0.167363 | 11.773 % | 84.01 % |
| | | kvarn2 + tail 1024 | 0.051440 | 6.597 % | 91.40 % |

What this says:

- **kvarn4 is effectively free on Codestral-22B**, the model actually served on
  port 8000: a mean KLD of 0.00135 with 98.4 % top-p agreement is well inside
  measurement noise for a 4-bit cache.
- **kvarn2 is where it breaks.** On Qwen2.5-VL-7B a mean KLD of 0.167 and 16 %
  top-p disagreement is a different model, not a cheaper cache. Do not use it
  without re-measuring on the real workload.
- **The 1024-token precision tail pays off most where quality is worst.** On the
  32B it improves kvarn4 from 0.0083 to 0.0012, a factor of seven, for 1024
  exact tokens per group. That is the intended trade and it is confirmed here.
- These numbers are single-corpus and four chunks. They separate "fine" from
  "broken" reliably; they do not rank two adjacent bit widths to three decimals.

## Fitness for autonomous agentic work

Measured after the tuning above, with the three production servers still
running. Short version: the inference substrate is fine, the agentic layer is
not there yet, and the machine cannot host one big model plus several small
services at the same time.

### Tool calling does not work with the models hosted here

`tools` requests sent to `/v1/chat/completions`, `tool_choice: auto`, one
`get_weather` function:

| Server | Result |
|---|---|
| Qwen2.5-VL-7B (port 8001) | no `tool_calls`; refuses and answers from memory |
| Codestral-22B (port 8000) | no `tool_calls`; **invents the tool result in prose** ("Tool: Weather / Input: Montreal / Output: The current weather is ...") |

Neither GGUF template mentions tools at all, checked through `/props` rather
than by grepping the file. A model that fabricates tool output is worse for an
agent loop than one that refuses, because the loop cannot tell.

`/v1/decision` covers part of the gap: tool selection is a classification, so
it works on a model with no tool template. Measured on the same Qwen2.5-VL-7B
that just refused, five routing requests over
`{read_file, grep_search, run_tests, git_diff, web_fetch, none}`: **5/5 correct**
at 64-69 ms, with probabilities from 0.87 to 1.00. The endpoint also fills
enum, boolean and bounded numeric arguments. It cannot produce free-form strings,
so file paths and search patterns still have to be generated by a model that
cannot reliably call tools in the first place.

### Generation does not stop where it should

On the Codestral server, `<|im_end|>` is emitted as text and generation
continues: a request with `max_tokens: 32` returned 32 tokens that ended in an
invented follow-up turn. Adding `"stop": ["<|im_end|>"]` to the request ends it
cleanly at 16 tokens. In a twenty-step agent loop this is twenty times the
intended compute plus fabricated turns in the transcript. Most clients send
their own stop sequences, which is why it stayed invisible.

### Decode throughput against context depth

Qwen2.5-14B Q4_K_M, tensor split, NCCL, `-ngl 99`, `-r 3`:

| Depth | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|
| 0 | 1658 | 59.6 |
| 4096 | 1347 | 55.8 |
| 8192 | 1176 | 52.4 |
| 16384 | 874 | 46.6 |

Decode loses 22 % by 16K, prefill loses 47 %. Prefill is what agentic turns
spend their time on, because tool output and file contents arrive as prompt.

### Speculative decoding helps one regime and hurts the other

Same code-rewrite prompt, 900 output tokens.

| Target | Placement | Draft | tg (t/s) |
|---|---|---:|---:|
| Qwen2.5-Coder-32B | auto-fit, part on CPU | none | 3.7 |
| Qwen2.5-Coder-32B | auto-fit, part on CPU | Qwen2.5-Coder-0.5B, n=6 | **11.7** |
| DeepSeek-R1-Distill-14B | all on GPU, tensor | none | 56.3 |
| DeepSeek-R1-Distill-14B | all on GPU, tensor | DeepSeek-1.5B, n=6 | 46.2 |

Drafting is a repair for CPU offload, not a general speedup: ×3.2 when the
weights do not fit, −18 % when they do. Check the pairing before trusting it —
a first attempt with Qwen2.5-0.5B against the DeepSeek 14B was rejected with
"the target and draft vocabs are not compatible" (BOS 151646 against 151643)
and silently measured the baseline twice.

N-gram speculation (`--spec-type ngram-simple`), which needs no draft model and
no VRAM, was neutral on this workload: 56.3 against 56.4 t/s. Rewriting code
does not repeat the prompt; verbatim copying might.

### What the machine can actually host

| Configuration | tg (t/s) | Verdict |
|---|---:|---|
| 14B all on GPU, tensor, 16K ctx, plus the three services | 46 | workable main loop |
| 32B sharing VRAM with the three services | 3.7 | unusable |
| 32B sharing VRAM plus a 0.5B draft | 11.7 | usable but slow |

The box runs one large model well or several small ones, not both. An agentic
deployment should give the model the machine: stop the auxiliary servers, raise
the context, and turn on KVarN to pay for that context inside the same VRAM.

