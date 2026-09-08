#!/bin/bash
# Watch the three ToF live: distance, the sensor's own verdict and the real stream rate.
# Wave a hand in front of one and its line changes immediately — that is the whole point.
#   ros/tof.sh          keep printing until Ctrl-C
# Status: 0 measured, 1 noisy, 2 signal too weak (usually nothing in range), 4 out of bounds,
# 5 hardware fail, 7 wraparound. A sensor stuck on 2 with something in front of it is blind.
set -uo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"
ssh "root@$BOARD" "python3 -c \"
import json, socket, time
s = socket.create_connection(('127.0.0.1', 3335))
f = s.makefile('r')
last, ticks, since = 0.0, 0, time.time()
rate = 0.0
for line in f:
    r = json.loads(line)
    ticks += 1
    if time.time() - since >= 1.0:
        rate, ticks, since = ticks / (time.time() - since), 0, time.time()
    if time.time() - last < 0.1:   # every reading the sensors make, not one a second
        continue
    last = time.time()
    st = r.get('status', {})
    print('  '.join(
        f'{n:>5}: ' + (f'{r[n]/1000:.2f} m' if r.get(n) is not None else '  --  ') + f' (status {st.get(n)})'
        for n in ('front', 'left', 'right')) + f'   {rate:5.1f} Hz', flush=True)
\""
