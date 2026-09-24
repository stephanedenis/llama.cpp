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

## gpt-oss-120b: the agentic model this machine can actually run

The models hosted before could not call tools at all. gpt-oss-120b can, and it
lives in RAM rather than VRAM, which plays to this host's strengths. Measured
with the three production servers stopped, 64 GB of experts in system RAM, and
attention plus KV on the two GPUs:

| Configuration | tg (t/s) | pp (t/s) | Note |
|---|---:|---:|---|
| `-cmoe`, mmap | 11.0 | 24 | reference |
| `-cmoe --no-mmap` | 10.4 | **34** | prefill +42 % |
| `-cmoe --no-mmap -ncmoe 24` | 11.1 | 35 | moving experts to GPU barely helps |
| `-cmoe --no-mmap` + EAGLE3 | **12.9** | 33 | draft accepted 594/915 = 65 % |

Native tool calling works: a `get_weather` request returns a proper
`tool_calls` entry with `{"city": "Montreal"}`, where Codestral invented the
result and Qwen2.5-VL refused.

`--no-mmap` is worth taking: llama.cpp warns that tensor overrides to CPU with
mmap enabled cost performance, and the prefill measurement agrees. It costs
about a minute of load time.

EAGLE3 is genuinely useful here (65 % acceptance, +24 % over the no-mmap
baseline), unlike draft-simple on the dense models, because verifying several
tokens amortises the same RAM read.

### Why this is the ceiling: memory bandwidth, not memory size

Measured sequential read bandwidth on this host:

| Placement | GB/s |
|---|---:|
| one socket, first-touch | 31.0 |
| both sockets, interleaved | **51.7** |
| a single core | 4.0 |

EDAC reports two DIMMs per socket, in channel 0 and channel 1. Haswell-EP has
**four** channels per socket, so half of the platform's memory channels are
empty. The measured 51.7 GB/s is consistent with four active channels of
DDR4-2133 at roughly 76 % efficiency.

That number governs CPU-resident inference. Decoding gpt-oss reads about 2.4 GB
of active expert weights per token; 2.4 GB against 52 GB/s is a hard ceiling
near 20 t/s, and the measured 11-13 t/s sits below it once attention and
scheduling are paid for. Halving the bandwidth halves the speed; filling the
empty channels would roughly double it.

So the upgrade advice is not "more RAM", it is "more channels":

- **512 GB as 8 x 64 GB** fills all eight channels: twice the bandwidth and four
  times the capacity. This is the configuration worth buying.
- **512 GB as 4 x 128 GB** adds capacity and no bandwidth: larger models at the
  same 11 t/s.
- Replacing the current 4 x 32 GB with 8 x 32 GB (256 GB) already buys the full
  bandwidth for half the money.

Capacity still matters, for a different reason: decode speed tracks the *active*
parameters, not the total, so 512 GB would allow a 200-400B sparse model with
around 5B active to answer at the same ~11 t/s. Capacity buys quality, bandwidth
buys speed, and VRAM still buys the only fast tokens.


### The memory upgrade, precisely

The host is a Dell Precision Tower 7910 (board 0215PR, BIOS A34) with 16 DIMM
slots, eight per socket. Four are populated today, two per socket, in channels 0
and 1; channels 2 and 3 are empty on both sockets. That is why the measured
bandwidth is half the platform's.

Dell lists these configurations for this chassis, all "DDR4 Registered":

| Fitted | Slots | DIMMs per channel | Speed |
|---|---:|---:|---|
| 128 GB (4 x 32 GB) | 4 of 16 | 1 (two sockets half filled) | DDR4-2133 |
| **256 GB (8 x 32 GB)** | 8 of 16 | 1, all channels | DDR4-2133 |
| **512 GB (8 x 64 GB)** | 8 of 16 | 1, all channels | DDR4-2133 |
| 512 GB (16 x 32 GB) | 16 of 16 | 2 | likely DDR4-1866 |

The second and third rows are the ones worth buying: eight DIMMs is one per
channel on all four channels of both sockets, which is full bandwidth at full
speed. Sixteen DIMMs doubles capacity again but puts two DIMMs on every channel,
which on Haswell-EP normally costs a speed step with dual-rank modules.

Expected gain: eight channels against four is 2x the theoretical bandwidth,
about 104 GB/s against the measured 51.7 GB/s. CPU-resident decoding is
bandwidth-bound, so gpt-oss-120b should move from 11-13 t/s towards 22-26 t/s.

What to buy: DDR4 ECC **Registered** (RDIMM), 288-pin, 1.2 V, PC4-2133P for
native speed. PC4-2400T or PC4-2666V also work and usually cost less; the E5 v3
memory controller runs them at 2133. Unbuffered ECC (UDIMM) will not post in
this machine. Match the existing modules' rank and vendor if adding to them
rather than replacing, and populate the slots in the order the owner's manual
specifies so both sockets stay balanced.

### Free optimisation: make both sockets carry the model

The DIMMs themselves are already at full speed: ~54 GB/s over four channels is
79 % of DDR4-2133's theoretical rate, which is what a healthy STREAM-style read
achieves. Rearranging four DIMMs cannot help either, because the total channel
count is what sets bandwidth, and it stays four wherever they are plugged.

What is free is making sure the weight buffer actually uses both sockets.
llama.cpp allocates the CPU-resident layers itself; without a NUMA policy the
pages land on whichever node first touches them, so one socket serves the whole
model and the other reaches across QPI.

Measured on gpt-oss-120b, experts in RAM, same prompt:

| Launch | tg (t/s) | pp (t/s) |
|---|---:|---:|
| default, no NUMA policy | 12.7 | 31 |
| **`numactl --interleave=all`** | **18.7** | 32 |
| `--numa distribute` (llama.cpp's own) | 12.5 | 32 |
| single socket, `--cpunodebind=0 --membind=0`, `-t 10` | 11.1 | 26 |

**+47 % decode for one wrapper command**, and the single-socket row confirms why:
confining the work to one node loses the other node's channels.

`scripts/serve-agentic.sh` applies this, and it is the whole win.

Two further knobs were tried and **measured to do nothing**. `optimize-system.sh`
briefly set both; the measurements below were taken with them active:

| Configuration | before | after |
|---|---:|---:|
| `numactl --interleave=all` | 18.7 t/s | 18.5 t/s |
| default, no policy | 12.7 t/s | 13.0 t/s |

Raw read bandwidth was unchanged too, 54.4-55.4 GB/s against 53.9-55.5 before.
The interleave gain itself reproduces (+42 %, 18.5 against 13.0), so the
measurement is sound; it is the two knobs that add nothing.

- `transparent_hugepage/defrag` does not help because the access pattern is
  streaming: each 4 KB page is consumed entirely before the next, so TLB misses
  amortise themselves. Huge pages pay off for sparse access, not for this.
- `numa_balancing` was already overridden by the explicit `numactl` policy, so
  turning it off changes nothing in the interleaved case.

`defrag` was left at `always` only for as long as it took to measure; the script
now leaves it alone, because `always` can stall allocations during compaction and
buys nothing here. `numa_balancing=0` is kept: it costs nothing and stops the
kernel from working against an explicit interleave policy.

### Why filling the empty slots with small modules does not pay

A tempting idea: buy four cheap small modules, move the four 32 GB ones onto
one socket, and end up with eight populated channels instead of four. The
channel arithmetic is right, the outcome is not, because bandwidth follows the
node that *holds* the data.

| Read pattern | GB/s |
|---|---:|
| memory on one node, local readers | 31.5 |
| memory on one node, readers on both sockets | **30.2** |
| memory interleaved, readers on both | **55.3** |

The middle row is the point: putting readers on the far socket does not add
bandwidth, it only adds latency. QPI carries traffic, it does not create
capacity. So a socket only contributes its channels in proportion to how much
of the working set it holds.

With 128 GB on one socket and 32 GB on the other, a 63 GB model splits roughly
43/20. Each controller can deliver about 55 GB/s, so the slower one sets the
pace: 43 GB at 55 GB/s, against 63 GB at 55 GB/s today. That is around 1.4x, not
2x. Balanced capacity is what buys the full doubling, because then each
controller carries half the reads: 31 GB each instead of 43.

Hence: four matched 32 GB modules, giving 128 GB per socket, rather than four
small ones. The imbalance costs most of the gain, and small DDR4 RDIMMs are not
on Dell's published list for this chassis (only 32 GB and 64 GB appear), so they
carry a compatibility risk on top.

Weighted interleave is available if an imbalanced layout ever has to be used:
the kernel is 7.2 and exposes `/sys/kernel/mm/mempolicy/weighted_interleave/`,
with `numactl --weighted-interleave`. It distributes pages in proportion to
capacity, but it cannot make a small socket carry a large share of the reads.

## Running without Internet

Verified rather than assumed: `scripts/check-offline.sh` re-runs the stack inside
an unprivileged network namespace (`unshare -rn`), which leaves loopback and
nothing else — no DNS, no route. It is the equivalent of unplugging the cable,
without touching the rest of the machine.

Result on this host, with the namespace confirmed sealed (no interface carrying
an address besides loopback, DNS unresolvable, HTTPS unreachable):

| Checked | Result |
|---|---|
| `llama-server` gpt-oss-120b + EAGLE3 | loads in 42 s, 16.8 t/s |
| native tool call | `{"name": "get_weather", "arguments": "{\"city\": \"Montreal\"}"}` |
| `llama-server` small model + `/v1/decision` | 48-90 ms |
| routing gateway, no external backend | serves, refuses egress by policy |

Nothing in the inference path reaches for the network. The build does not either:
third-party code is vendored under `vendor/`, and the one `FetchContent` in
`ggml-cuda` is behind `GGML_CUDA_CUB_3DOT2`, which is off, so a rebuild from a
clean tree needs no download.

### What still talks to the outside

The inference stack is clean, but two neighbouring services are not, and both
matter for a real air gap:

- **open-webui** runs in Docker with working outbound access, and its
  `OFFLINE_MODE` defaults to `false`, which leaves the version update check on.
  Anonymous telemetry is already disabled here. Set `OFFLINE_MODE=true` (it also
  sets `HF_HUB_OFFLINE=1` and turns the update check off) before treating the box
  as isolated.
- **ollama** is running with about 44 GB of models that duplicate what
  llama.cpp already serves, and it can pull more on demand. If it is not in use,
  stop and disable it rather than leaving a second network-capable model server
  on the machine.

Model downloads themselves are the obvious one-time exception: every model in
use is already on disk under `~/FastNVMe/models`.
