#!/usr/bin/env bash
# Lance notre llama-server (build CUDA) avec les réglages optimaux
# pour le serveur : 2× Xeon E5-2687W v3 + 2× Quadro RTX 5000 (NVLink NV1)
#
# Usage :
#   ./scripts/serve.sh <model.gguf> [port] [options supplémentaires...]
#
# Exemples :
#   ./scripts/serve.sh model.gguf                     # port 8080, GPU offload total
#   ./scripts/serve.sh model.gguf 9999 -c 32768       # port 9999, contexte 32k
#   ./scripts/serve.sh moe.gguf 8080 -ncmoe 26        # gros MoE offload
#
# Mode tensor (expérimental, +54 % de génération mesuré sur ce matériel) :
#   LLAMA_SPLIT_MODE=tensor ./scripts/serve.sh model.gguf

set -e
MODEL="$1"
PORT="${2:-8080}"
shift 2 2>/dev/null || shift

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Usage : $0 <model.gguf> [port] [options...]"
    echo ""
    echo "Modèles GGUF disponibles :"
    find ~/FastNVMe/models ~/Data13TB/stephane/models -name "*.gguf" 2>/dev/null | head -20
    exit 1
fi

# Nombre de threads hôte.
#
# ATTENTION — ne pas remettre -t 40 : mesuré sur Qwen2.5-14B Q4_K_M avec
# offload total, « -t 40 --numa distribute » fait chuter le prompt processing
# à 163 t/s contre 1090 t/s avec -t 20 sans --numa (facteur 6,7). Le CPU ne
# fait ici que l'échantillonnage et la préparation de batch : quelques
# threads d'un seul nœud NUMA suffisent, et --numa distribute est inutile
# dès que les poids sont sur GPU.
THREADS="${LLAMA_THREADS:-20}"

# layer (défaut) ou tensor. Le mode tensor répartit poids ET KV sur les deux
# GPU : mesuré à 59,6 t/s en génération contre 38,6 en layer, grâce au NVLink.
# Il reste marqué expérimental en amont (docs/multi-gpu.md) : valider la
# qualité des sorties avant de le déployer.
SPLIT_MODE="${LLAMA_SPLIT_MODE:-layer}"

exec llama-server \
    -m "$MODEL" \
    -ngl 99 \
    -t "$THREADS" \
    -sm "$SPLIT_MODE" \
    --flash-attn on \
    --port "$PORT" \
    "$@"
