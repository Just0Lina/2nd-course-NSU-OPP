#!/bin/bash

i=1
while [ "$i" -le 12 ]
do
echo "$i threads" | tr -s '\r\n' ' '>> lab2/fast/fast.txt
OMP_NUM_THREADS=$i ./openMp | tr -s '\r\n' ' ' >> lab2/fast/fast.txt
OMP_NUM_THREADS=$i ./openMp >> lab2/fast/fast.txt
i=$(( i + 1 ))
done
