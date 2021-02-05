#!/bin/bash

for i in 4 5 6 7; do
    echo "Getting h2_${i} data"
    grep -e "T2 ratio" -e "Tcut_PNO" h2_$i.dat | cut -f2- -d: | awk '!visited[$0]++' | awk '{ sub(/^[ ]+/, ""); print }'| awk -v dq="\"" 'BEGIN{print "{"} {if (NR%2==0) {print $0","} else {print dq$0dq ":"} }' >> h2_${i}_t2.json
    grep -e "T2 ratio" -e "Tcut_PNO" h2_${i}_more.dat | cut -f2- -d: | awk '!visited[$0]++' | awk '{ sub(/^[ ]+/, ""); print }'| awk -v dq="\"" '{if (NR%2==0) {print $0","} else {print dq$0dq ":"} } END { print "}" }' >> h2_${i}_t2.json

done

exit 0
