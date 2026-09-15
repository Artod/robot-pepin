#!/bin/bash
# Shared by the ros/*.sh scripts: one multiplexed ssh connection to the board.
# The first handshake costs 2.5-3.5 s here (sshd + PAM on a loaded A53, measured 2026-09-06);
# every later command over the same master costs ~0.1 s. Source after BOARD is set;
# `ssh`, `scp` and rsync then reuse the master for 10 minutes of idle.
BOARD="${BOARD:-${PEPIN_HOST:-10.0.0.187}}"
PEPIN_SSH_OPTS="-o ControlMaster=auto -o ControlPath=/tmp/pepin-ssh-%C -o ControlPersist=10m -o ConnectTimeout=6"
ssh() { command ssh $PEPIN_SSH_OPTS "$@"; }
scp() { command scp $PEPIN_SSH_OPTS "$@"; }
export RSYNC_RSH="ssh $PEPIN_SSH_OPTS"

# The one way to stop a container here; no ros/*.sh calls `docker stop`, `docker kill` or
# `docker rm -f` on its own. `docker stop` sends the container's STOPSIGNAL, which is SIGINT for
# everything built from ros/Dockerfile (and passed again as --stop-signal by the docker run
# lines, so a container started from another image is gentle too): SIGINT is the signal ros2
# launch answers by shutting its nodes down, while SIGTERM it answers by cancelling itself and
# the nodes are SIGKILLed mid-write. That difference is how ros/maps/rtabmap.db was made
# malformed (2026-09-13, eight kills of a crash loop; a torn page read on 2026-09-15).
# The window is pepin.deployment.CONTAINER_STOP_TIMEOUT_S seconds — RTAB-Map closing 20-28 GB of
# visual memory is the slowest thing in it — and docker SIGKILLs at its end, which is why the
# database is also configured to survive a kill (DbSqlite3/JournalMode in vslam.launch.py).
PEPIN_STOP_TIMEOUT_S="${PEPIN_STOP_TIMEOUT_S:-30}"  # = pepin.deployment.CONTAINER_STOP_TIMEOUT_S
pepin_stop_container() {  # NAME...: stop gently, leave the stopped container (a unit keeps its log)
    [ "$#" -gt 0 ] || return 0
    docker stop -t "$PEPIN_STOP_TIMEOUT_S" "$@" >/dev/null 2>&1 || true
}
pepin_remove_container() {  # NAME...: stop gently, then remove — a container about to be replaced
    pepin_stop_container "$@"
    docker rm -f "$@" >/dev/null 2>&1 || true
}

# What the robot is thinking, printed inline while a script drives it: goals, planner and
# controller verdicts, recoveries (spin/backup/wait), AMCL and relocalizer lines, local time.
# watch_start once before driving, watch_stop at the end (ros/watch.sh is the same view alone).
PEPIN_WATCH_KEEP='bt_navigator|behavior_server|controller_server|planner_server|amcl\]|relocalizer\]|velocity_smoother'
PEPIN_WATCH_DROP='Passing new path|foxglove|Message Filter|bond|Load Library|Found class|Instantiate class|launch_ros|lifecycle node launched|: Creating|: Configuring|: Activating|: Cleaning|Original Node|Setting|\]: $'
pepin_render() {  # "[proc] [LEVEL] [epoch] [node]: text" -> "HH:MM:SS LEVEL node: text"; repeats folded; \r\n so a pty next door cannot stair-step it
    perl -MPOSIX=strftime -ne '
        BEGIN { $| = 1; $last = ""; $n = 0 }
        chomp;
        next if /\[ActionServer\]|begin computing control effort|Received a goal|Client requested to cancel|Cancellation was successful/;
        s/RegulatedPurePursuitController/RPP/; s/GridBased plugin //; s/\[nav2_behaviors\] //;
        if (s/^\[[^\]]*\] \[(INFO|WARN|ERROR|DEBUG)\] \[(\d{10})\.\d+\] \[([^\]]+)\]: /sprintf("%s %-5s %s: ", strftime("%H:%M:%S", localtime($2)), $1, $3)/e) {
            ($msg = $_) =~ s/^\S+ //;
            if ($msg eq $last) { $n++; next }
            print "\r         ... x$n more\r\n" if $n;
            $n = 0; $last = $msg;
        }
        print "\r$_\r\n";
    '
}
watch_start() {
    # A viewer is for a person at a terminal. Run with its output piped (an agent, a script),
    # it has no reader, and every way of stopping it has left some member alive holding the
    # pipe open, so the caller never returned (2026-09-13, three times): no terminal, no viewer.
    [ -t 1 ] || return 0
    # The viewer runs in a process group of its own, so the stop reaches every member — the
    # ssh streaming the log included — with one signal. Killing the subshell alone, or pkill
    # by command line, left the ssh alive; it held the caller's stdout open, and a script whose
    # goto output was piped never returned (2026-09-13: the cart stood at the printer for nine
    # minutes while goto.sh waited on its own viewer). python's setpgrp, not bash's `set -m`:
    # job control needs a terminal, and a script run with its stdin on /dev/null got no group.
    export -f pepin_render
    python3 -c 'import os, sys; os.setpgrp(); os.execvp(sys.argv[1], sys.argv[1:])' bash -c \
        'ssh "root@$0" "docker logs -f --since 3s pepin-ros 2>&1" | grep --line-buffered -E "$1" | grep --line-buffered -vE "$2" | pepin_render' \
        "$BOARD" "$PEPIN_WATCH_KEEP" "$PEPIN_WATCH_DROP" 2>/dev/null &
    PEPIN_WATCH_PID=$!
    disown "$PEPIN_WATCH_PID" 2>/dev/null || true
}
watch_stop() {  # never blocks: a stuck viewer must not delay the stop that follows it
    [ -n "${PEPIN_WATCH_PID:-}" ] || return 0
    kill -TERM -- "-$PEPIN_WATCH_PID" 2>/dev/null || kill "$PEPIN_WATCH_PID" 2>/dev/null
    PEPIN_WATCH_PID=""
}
