#!/usr/bin/env bats

PCAP="/home/vlad/Desktop/Probleme/E3S/CIC-IDS2017/PCAPs/Friday-WorkingHours.pcap"
CFG="/home/vlad/Documents/SublimeMerge/snort3_demo/tests/search_engines/bruteforce/snort.lua"
OPTION="-q -A csv -k none"
DAQDIR="/usr/local/lib/daq"
PLUGINPATH="/home/vlad/Documents/SublimeMerge/snort3_extra/build/src"

@test "bruteforce search method" {
    $snort -r $PCAP -c $CFG --plugin-path $PLUGINPATH --daq-dir $DAQDIR $OPTION > snort.out
    diff expected snort.out
}


