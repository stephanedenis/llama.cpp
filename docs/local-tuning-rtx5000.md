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
