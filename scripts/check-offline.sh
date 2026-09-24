#!/usr/bin/env bash
# Verifie que la pile IA locale fonctionne sans aucun acces Internet.
#
# La methode : un namespace reseau isole (unshare -rn), donc aucune interface
# autre que loopback, pas de DNS, pas de route. C'est equivalent a debrancher le
# cable, sans toucher au reste de la machine ni avoir besoin de root.
#
# Usage :
#   ./scripts/check-offline.sh            # test rapide : petit modele + /v1/decision
#   ./scripts/check-offline.sh --full     # ajoute gpt-oss-120b et un appel d'outil
#
# Le test rapide prend ~15 s, le test complet ~3 min (chargement de 63 Go).

set -u

FULL=""
[ "${1:-}" = "--full" ] && FULL=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${LLAMA_MODELS_DIR:-$HOME/FastNVMe/models}"
SMALL="$MODELS/qwen2.5-1.5b-instruct-q4_k_m.gguf"
BIG="$MODELS/gpt-oss-120b-MXFP4.gguf"
DRAFT="$MODELS/eagle3-gpt-oss-120b-Q8_0.gguf"

command -v unshare >/dev/null || { echo "unshare absent" >&2; exit 1; }
unshare -rn true 2>/dev/null || { echo "namespace reseau non privilegie refuse" >&2; exit 1; }

cat > /tmp/check-offline-inner.sh <<'INNER'
#!/usr/bin/env bash
set -u
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
cd "$ROOT" || exit 1
BIN=./build-cuda/bin/llama-server

echo "--- etancheite du namespace"
ip link set lo up 2>/dev/null
n_if=$(ip addr show 2>/dev/null | grep -c "inet ")
echo "    interfaces avec adresse IP : $n_if  (1 = loopback seul, attendu)"
timeout 3 getent hosts huggingface.co >/dev/null 2>&1 \
    && echo "    DNS      : RESOLUTIF  <-- anormal" \
    || echo "    DNS      : injoignable (attendu)"
timeout 5 curl -s -o /dev/null https://example.com 2>/dev/null \
    && echo "    HTTPS    : joignable   <-- anormal" \
    || echo "    HTTPS    : injoignable (attendu)"

fail=0
wait_health() {
    local port=$1 pid=$2
    for _ in $(seq 1 150); do
        sleep 2
        curl -s -m 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q ok && return 0
        kill -0 "$pid" 2>/dev/null || return 1
    done
    return 1
}

echo "--- couche reflexe : petit modele + /v1/decision"
$BIN -m "$SMALL" -ngl 99 -fa on -t 8 -c 4096 --decision-seqs 16 \
    --host 127.0.0.1 --port 8082 > /tmp/offline-small.log 2>&1 &
REF=$!
if wait_health 8082 $REF; then
    curl -s -m 120 http://127.0.0.1:8082/v1/decision -H "Content-Type: application/json" -d '{
      "contexts":["Show me the diff of my uncommitted changes."],
      "instructions":"You dispatch a coding agent to one tool.",
      "schema":{"tool":{"type":"enum","description":"Which tool to call.",
                        "choices":["read_file","grep_search","git_diff","none"]}}}' \
      | python3 -c "
import json,sys
d=json.load(sys.stdin); r=d['results'][0]
print('    /v1/decision :', r['decision']['tool'],
      '| p =', round(r['fields']['tool']['probability'],2),
      '|', round(d['timings']['per_decision_ms']), 'ms')
" || fail=1
else
    echo "    ECHEC : serveur non demarre"; fail=1
fi
kill $REF 2>/dev/null; sleep 3; wait $REF 2>/dev/null

if [ -n "$FULL" ]; then
    echo "--- cerveau agentique : gpt-oss-120b + EAGLE3 + appel d'outil"
    if [ ! -f "$BIG" ] || [ ! -f "$DRAFT" ]; then
        echo "    modeles absents, test ignore"
    else
        numactl --interleave=all $BIN -m "$BIG" -fa on -c 8192 -ngl 99 -cmoe --no-mmap \
            --spec-type draft-eagle3 -md "$DRAFT" -ngld 99 -t 20 \
            --host 127.0.0.1 --port 8081 > /tmp/offline-big.log 2>&1 &
        BRAIN=$!
        if wait_health 8081 $BRAIN; then
            curl -s -m 300 http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" -d '{
              "messages":[{"role":"user","content":"What is the weather in Montreal? Use the tool."}],
              "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
              "max_tokens":200,"temperature":0}' | python3 -c "
import json,sys
d=json.load(sys.stdin); m=d['choices'][0]['message']; t=d.get('timings',{})
tc=m.get('tool_calls')
print('    tool_calls :', json.dumps(tc[0]['function']) if tc else 'AUCUN')
print('    debit      :', round(t.get('predicted_per_second',0),1), 't/s')
if not tc: sys.exit(1)
" || fail=1
        else
            echo "    ECHEC : serveur non demarre"; fail=1
        fi
        kill $BRAIN 2>/dev/null; sleep 5; wait $BRAIN 2>/dev/null
    fi
fi

echo ""
[ $fail -eq 0 ] && echo "RESULTAT : la pile fonctionne hors ligne" \
                || echo "RESULTAT : ECHEC, voir /tmp/offline-*.log"
exit $fail
INNER
chmod +x /tmp/check-offline-inner.sh

ROOT="$ROOT" FULL="$FULL" SMALL="$SMALL" BIG="$BIG" DRAFT="$DRAFT" \
    unshare -rn env ROOT="$ROOT" FULL="$FULL" SMALL="$SMALL" BIG="$BIG" DRAFT="$DRAFT" \
    /tmp/check-offline-inner.sh
