# The kicked node's successor and its own ready line, read from its container's log (the tail
# `docker logs -t --since <the kick>` prints). ros/board.sh kick and ros/laptop.sh kick feed it
# on every poll; POSIX awk (mawk on the board, BWK awk on the Mac).
#
# The launch writes three lines per restart under the process's own tag ([python3-7]):
#   [ERROR] [python3-7]: process has died [pid 47, exit code 1, cmd '...']   (or finished cleanly)
#   [INFO] [python3-7]: process started with pid [836]
#   [python3-7] [INFO] [1790206544.19] [bag_recorder]: bag recorder ready: ...
# The first names the tag of the pid that was sent SIGINT; only a line under that tag AFTER the
# successor's start can be the new process's, so an old ready line is never read as the new one.
#
# In:  -v name=NODE -v old="PID..." (the pids sent SIGINT) -v line="its ready text"
#      -v kicked=<RFC3339 time of the SIGINT, the container's clock>
# Out: one line, "ready<TAB>NODE kicked at ..., ready at ..., S s, pid OLD -> NEW: TEXT", or
#      "wait<TAB>what is still missing".
function tag_of(msg, p,    head, i) {  # the tag before "]: " at p: "[INFO] [python3-7]:" -> python3-7
    head = substr(msg, 1, p - 1)
    for (i = length(head); i > 0; i--)
        if (substr(head, i, 1) == "[") return substr(head, i + 1)
    return ""
}
function pid_after(msg, key,    q, rest) {  # the digits right after key, or ""
    q = index(msg, key)
    if (q == 0) return ""
    rest = substr(msg, q + length(key))
    if (match(rest, /^[0-9]+/)) return substr(rest, 1, RLENGTH)
    return ""
}
function sod(ts,    h, m, s) {  # seconds of the UTC day of an RFC3339 time
    h = substr(ts, 12, 2); m = substr(ts, 15, 2); s = substr(ts, 18)
    sub(/Z$/, "", s)
    return h * 3600 + m * 60 + s
}
function clock(ts) {  # "2026-09-27T10:00:05.123456789Z" -> "10:00:05.1"
    if (ts == "") return "?"
    return substr(ts, 12, 8) (substr(ts, 20, 1) == "." ? substr(ts, 20, 2) : "")
}
BEGIN {
    n = split(old, pids, " ")
    for (i = 1; i <= n; i++) olds[pids[i]] = 1
    stage = "exit"; tag = ""; newpid = ""; crashes = 0; done = 0
}
{
    ts = ""; msg = $0
    if ($1 ~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T/) { ts = $1; msg = substr($0, length($1) + 2) }
    if (stage == "exit") {
        p = index(msg, "]: process has ")
        if (p > 0) {
            pid = pid_after(msg, "[pid ")
            if (pid in olds) { tag = tag_of(msg, p); gone = pid; stage = "start" }
        }
        next
    }
    if (stage == "start") {
        p = index(msg, "]: process started with pid [")
        if (p > 0 && tag_of(msg, p) == tag) { newpid = pid_after(msg, "with pid ["); stage = "ready" }
        next
    }
    p = index(msg, "]: process has ")
    if (p > 0 && tag_of(msg, p) == tag) { crashes++; stage = "start"; next }
    prefix = "[" tag "] "
    if (substr(msg, 1, length(prefix)) == prefix && index(msg, line) > 0) {
        text = substr(msg, length(prefix) + 1)
        q = index(text, "]: ")
        if (q > 0) text = substr(text, q + 3)
        secs = (ts != "" && kicked != "") ? sprintf("%.1f s", (sod(ts) - sod(kicked) + 86400) % 86400) : "? s"
        printf "ready\t%s kicked at %s, ready at %s UTC, %s, pid %s -> %s: %s\n", name, clock(kicked), clock(ts), secs, gone, newpid, text
        done = 1
        exit
    }
}
END {
    if (done) exit
    if (stage == "exit")
        printf "wait\tpid %s has not exited since the SIGINT (a node that ignores it needs kill -9; the launch respawns it)\n", old
    else if (stage == "start")
        printf "wait\t%s (pid %s) exited%s; no successor started yet (the launch respawns 2 s after an exit)\n", tag, gone, crashes ? ", and its successor died " crashes " time(s)" : ""
    else
        printf "wait\tpid %s (%s) started but has not printed '%s' yet\n", newpid, tag, line
}
