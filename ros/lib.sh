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
    ( ssh "root@$BOARD" "docker logs -f --since 3s pepin-ros 2>&1" \
        | grep --line-buffered -E "$PEPIN_WATCH_KEEP" | grep --line-buffered -vE "$PEPIN_WATCH_DROP" | pepin_render ) 2>/dev/null &
    PEPIN_WATCH_PID=$!
    disown "$PEPIN_WATCH_PID" 2>/dev/null || true
}
watch_stop() {  # never blocks: a stuck viewer must not delay the stop that follows it
    [ -n "${PEPIN_WATCH_PID:-}" ] || return 0
    # Kill the ssh that streams the log by its command line as well as by pid: it is a grandchild
    # of this shell, and killing the subshell alone left it printing into the terminal for minutes.
    { pkill -P "$PEPIN_WATCH_PID"; kill "$PEPIN_WATCH_PID"; } 2>/dev/null
    pkill -f "docker logs -f --since 3s pepin-ros" 2>/dev/null
    PEPIN_WATCH_PID=""
}
