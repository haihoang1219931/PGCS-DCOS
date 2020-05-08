echo 1 | sudo -S ip route add 224.0.0.0/4 dev enx0022201d5bba
echo 1 | sudo -S ip route replace default via 192.168.43.1 dev wlp1s0 proto static
cd /home/pgcs-05/PGCS-DCOS/Exe && ./PGCS-DCOS_v2_Experiment
