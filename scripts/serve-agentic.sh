#!/usr/bin/env bash
# Serveur agentique : gpt-oss-120b avec les experts en RAM.
#
# Pourquoi cette configuration plutot que celle de serve.sh :
#
#   * gpt-oss-120b (117B total, 5,1B actifs) est le seul modele teste ici qui
#     fasse des appels d'outils natifs corrects. Codestral en inventait le
#     resultat, Qwen2.5-VL refusait.
#   * Le modele ne tient pas en VRAM (63 Go pour 30 Go disponibles), donc les
#     experts vont en RAM. L'attention et le KV restent sur les GPU.
#   * La RAM est le goulot : ~54 GB/s mesures. Repartir les poids sur les DEUX
#     sockets via numactl --interleave=all fait passer le decodage de 12,7 a
#     18,7 t/s (+47 %). Sans politique NUMA, tout atterrit sur un seul noeud et
#     la moitie de la bande passante dort.
#   * --no-mmap gagne 42 % en prefill (24 -> 34 t/s) : llama.cpp previent lui-meme
#     que mmap coute cher quand des tenseurs sont forces vers le CPU.
#   * EAGLE3 est utile ici (65 % d'acceptation, +24 %) : verifier plusieurs
#     tokens amortit une seule lecture des experts en RAM.
#
# Usage : ./scripts/serve-agentic.sh [port] [options supplementaires...]

set -e

MODEL="${LLAMA_AGENTIC_MODEL:-$HOME/FastNVMe/models/gpt-oss-120b-MXFP4.gguf}"
DRAFT="${LLAMA_AGENTIC_DRAFT:-$HOME/FastNVMe/models/eagle3-gpt-oss-120b-Q8_0.gguf}"
PORT="${1:-8010}"
shift 2>/dev/null || true

for f in "$MODEL" "$DRAFT"; do
    [ -f "$f" ] || { echo "introuvable : $f" >&2; exit 1; }
done

# Les poids sont en RAM : les deux sockets doivent y contribuer.
# --interleave=all repartit chaque allocation sur les deux noeuds NUMA.
NUMA="numactl --interleave=all"
command -v numactl >/dev/null || { echo "numactl absent, le decodage perdra ~47 %" >&2; NUMA=""; }

exec $NUMA llama-server \
    -m "$MODEL" \
    -ngl 99 \
    -cmoe \
    --no-mmap \
    -fa on \
    -t 20 \
    -c 16384 \
    --spec-type draft-eagle3 \
    -md "$DRAFT" \
    -ngld 99 \
    --host 127.0.0.1 \
    --port "$PORT" \
    "$@"
