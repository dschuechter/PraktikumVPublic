#!/bin/bash
a=0
b=1000
while [ $b -le 1001 ]; do
#cmd="python3 zusatz.py $a $b";
termite -e 'python3 zusatz.py $a $b' &
#"${cmd}" &>/dev/null & disown;
a=$(($a+100));
b=$(($b+100));
done
