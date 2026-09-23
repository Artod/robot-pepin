# The laptop's time server: chronyd serving THIS Docker VM's clock — the clock every ROS node on
# the laptop stamps with, since all containers share the VM's one kernel — to the board over NTP
# (board/chrony.sh points the board's chrony at it). Built once by ros/lib.sh's
# pepin_timeserver_up; the configuration is mounted from ros/chrony/laptop.conf, so editing it
# needs a container restart, never a rebuild.
FROM alpine:3.20
RUN apk add --no-cache chrony
EXPOSE 123/udp
# -d: foreground, messages to `docker logs`. -x: never adjust the clock — this container holds no
# CAP_SYS_TIME and serves the VM's time exactly as the ROS nodes read it (laptop.conf says why).
ENTRYPOINT ["chronyd", "-d", "-x", "-f", "/etc/chrony/chrony.conf"]
