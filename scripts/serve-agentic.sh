#!/usr/bin/env bash
# Serveur agentique : deux etages.
#
#   1. le cerveau (gpt-oss-120b) — raisonnement et appels d'outils natifs
#   2. le reflexe (petit modele + /v1/decision) — decisions structurees en
#      quelques dizaines de millisecondes, sans generer de texte
#
# Pourquoi cette configuration plutot qu'une seule :
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
#   * Le reflexe repond en 55-80 ms la ou le cerveau mettrait des secondes a
#     produire le meme arbitrage, et il rend une probabilite par champ avec un
#     JSON assemble par du code, donc jamais malforme.
#
# Usage : ./scripts/serve-agentic.sh [port_cerveau] [options...]
#   --no-reflex   ne pas demarrer la couche reflexe
#
# Variables :
#   LLAMA_AGENTIC_MODEL / LLAMA_AGENTIC_DRAFT   cerveau et son brouillon
#   LLAMA_REFLEX_MODEL                          modele de la couche reflexe
#   AGENTIC_REFLEX_PORT                         port du reflexe (defaut 8011)

set -e

PORT="${1:-8010}"
shift 2>/dev/null || true

NO_REFLEX=""
ARGS=()
for a in "$@"; do
    if [ "$a" = "--no-reflex" ]; then NO_REFLEX=1; else ARGS+=("$a"); fi
done

MODELS="${LLAMA_MODELS_DIR:-$HOME/FastNVMe/models}"
MODEL="${LLAMA_AGENTIC_MODEL:-$MODELS/gpt-oss-120b-MXFP4.gguf}"
DRAFT="${LLAMA_AGENTIC_DRAFT:-$MODELS/eagle3-gpt-oss-120b-Q8_0.gguf}"
REFLEX_PORT="${AGENTIC_REFLEX_PORT:-8011}"

for f in "$MODEL" "$DRAFT"; do
    [ -f "$f" ] || { echo "introuvable : $f" >&2; exit 1; }
done

# Modele de la couche reflexe. La precision depend beaucoup de la taille, et
# c'est mesure sur ce jeu d'evaluation :
#   Qwen2.5-VL-7B  95,8 % de sensibilite, 1 erreur dangereuse sur 24, 62 ms
#   Qwen2.5-1.5B   62,5 %, 3 erreurs dangereuses sur 24, 34 ms
# Preferer le 7B : un classifieur rapide et faux ne sert a rien ici.
REFLEX=""
if [ -n "${LLAMA_REFLEX_MODEL:-}" ]; then
    REFLEX="$LLAMA_REFLEX_MODEL"
else
    for c in "$MODELS/qwen2.5-7b-instruct-q4_k_m.gguf" \
             "$MODELS/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" \
             "$MODELS/qwen2.5-1.5b-instruct-q4_k_m.gguf"; do
        [ -f "$c" ] && { REFLEX="$c"; break; }
    done
fi
[ -z "$NO_REFLEX" ] && [ -z "$REFLEX" ] && {
    echo "aucun modele reflexe trouve, demarrage sans la couche reflexe" >&2
    NO_REFLEX=1
}

# Les poids sont en RAM : les deux sockets doivent y contribuer.
NUMA="numactl --interleave=all"
command -v numactl >/dev/null || { echo "numactl absent, le decodage perdra ~47 %" >&2; NUMA=""; }

PIDS=()
cleanup() {
    trap - INT TERM EXIT
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# ------------------------------------------------------------------ reflexe
if [ -z "$NO_REFLEX" ]; then
    echo "reflexe : $(basename "$REFLEX") sur 127.0.0.1:$REFLEX_PORT"
    llama-server \
        -m "$REFLEX" \
        -ngl 99 \
        -fa on \
        -t 8 \
        -c 4096 \
        --decision-seqs 16 \
        --host 127.0.0.1 \
        --port "$REFLEX_PORT" > /tmp/serve-agentic-reflex.log 2>&1 &
    REFLEX_PID=$!
    PIDS+=("$REFLEX_PID")
fi

# ------------------------------------------------------------------- cerveau
echo "cerveau : $(basename "$MODEL") sur 127.0.0.1:$PORT"
$NUMA llama-server \
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
    "${ARGS[@]+"${ARGS[@]}"}" > /tmp/serve-agentic-brain.log 2>&1 &
BRAIN_PID=$!
PIDS+=("$BRAIN_PID")

# --------------------------------------------------------------- attente
wait_health() {
    local port=$1 pid=$2 name=$3
    for _ in $(seq 1 180); do
        sleep 2
        curl -s -m 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q ok && return 0
        kill -0 "$pid" 2>/dev/null || { echo "  $name : processus mort" >&2; return 1; }
    done
    echo "  $name : pas de reponse apres 360 s" >&2
    return 1
}

ok=1
[ -n "$NO_REFLEX" ] || wait_health "$REFLEX_PORT" "$REFLEX_PID" "reflexe" || ok=0
wait_health "$PORT" "$BRAIN_PID" "cerveau" || ok=0
[ "$ok" = 1 ] || exit 1

# --------------------------------------------------- verification du reflexe
# Deux appels : le premier paie le prefill des instructions et du catalogue de
# champs (279 ms mesures), le second donne ce que coutera une decision reelle.
# On prechauffe donc ici, et on rapporte la latence a chaud.
if [ -z "$NO_REFLEX" ]; then
    decision() {
        curl -s -m 60 "http://127.0.0.1:$REFLEX_PORT/v1/decision" \
            -H "Content-Type: application/json" \
            -d '{"contexts":["Show me the diff of my uncommitted changes."],
                 "instructions":"You dispatch a coding agent to one tool.",
                 "schema":{"tool":{"type":"enum","description":"Which tool to call.",
                                   "choices":["read_file","grep_search","git_diff","none"]}}}'
    }
    decision > /dev/null 2>&1 || true
    decision 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); r=d['results'][0]
    print('  /v1/decision :', r['decision']['tool'],
          '| p =', round(r['fields']['tool']['probability'], 2),
          '|', round(d['timings']['per_decision_ms']), 'ms (a chaud)')
except Exception as e:
    print('  /v1/decision : pas de reponse exploitable :', e)
" || echo "  /v1/decision : verification impossible"
fi

cat <<EOF

pret.
  cerveau  http://127.0.0.1:$PORT/v1      (chat, appels d'outils natifs)
$( [ -z "$NO_REFLEX" ] && echo "  reflexe  http://127.0.0.1:$REFLEX_PORT/v1/decision   (decisions structurees)" )
  journaux /tmp/serve-agentic-{brain,reflex}.log

Ctrl-C pour tout arreter.
EOF

# On reste au premier plan : le trap tue les deux serveurs a l'arret.
wait
