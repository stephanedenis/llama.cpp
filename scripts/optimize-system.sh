#!/usr/bin/env bash
# Optimisation système pour inférence llama.cpp
# À exécuter avec : sudo ./scripts/optimize-system.sh
#
# Cible : 2x Xeon E5-2687W v3 (Haswell, AVX2), 2x Quadro RTX 5000, 125 GB RAM

set -e

echo "=== 1. CPU governor -> performance ==="
# schedutil -> performance : évite la montée en fréquence progressive,
# utile pour les inférences longues (prompt eval soutenu)
if command -v cpupower >/dev/null 2>&1; then
    cpupower frequency-set -g performance
else
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$f" 2>/dev/null || true
    done
fi
echo "  Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"

echo "=== 2. Transparent Huge Pages -> always ==="
# THP "always" réduit les défauts de page pour les gros tensors mmapés
echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo "  THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled | head -1)"

# defrag : MESURÉ SANS EFFET sur ce matériel. Passer defrag à "always" pour
# former des pages de 2 Mo n'a rien changé, ni au débit mémoire brut
# (54,4-55,4 GB/s contre 53,9-55,5) ni au décodage de gpt-oss (18,5 contre
# 18,7 t/s). L'accès est séquentiel : chaque page de 4 Ko est lue en entier
# avant la suivante, donc les défauts de TLB s'amortissent seuls. "always" peut
# en plus provoquer des à-coups pendant la compaction. On n'y touche pas.
echo "  THP defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag | head -1) (laissé tel quel)"

# NUMA balancing : également sans effet mesurable, mais conservé car il ne coûte
# rien et empêche le noyau de défaire une politique d'entrelacement explicite.
echo 0 > /proc/sys/kernel/numa_balancing 2>/dev/null || true
echo "  numa_balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null)"

echo "=== 3. swappiness -> 10 ==="
# Avec 125 GB RAM, on évite de swapper prématurément
sysctl -w vm.swappiness=10 2>/dev/null || true

echo "=== 4. vfs_cache_pressure -> 50 ==="
# Garde plus de cache inode (utile pour recharger des modèles)
sysctl -w vm.vfs_cache_pressure=50 2>/dev/null || true

echo "=== 5. Verification NUMA ==="
numactl --hardware | head -4

echo ""
echo "=== Fait. Pour appliquer de façon permanente, ajouter dans /etc/sysctl.d/99-llama.conf :"
cat <<'EOF'
vm.swappiness=10
vm.vfs_cache_pressure=50
vm.max_map_count=1048576
EOF
echo ""
echo "=== Et pour le governor permanent :"
echo "  installer 'cpupower' et activer le service 'cpupower' avec governor=performance"
