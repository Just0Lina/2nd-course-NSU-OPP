#!/bin/bash
# mpic++ mpi.cpp  -o mpi   
i=1
while [ "$i" -le 12 ]
do
echo "$i threads" | tr -s '\r\n' ' '>> result.txt
mpiexec -n $i ./mpi    | tr -s '\r\n' ' ' >>  result.txt
mpiexec -n $i ./mpi  | tr -s '\r\n' ' ' >>  result.txt
mpiexec -n $i ./mpi | tr -s '\r\n' ' ' >>  result.txt
mpiexec -n $i ./mpi | tr -s '\r\n' ' ' >>  result.txt
mpiexec -n $i ./mpi  >>  result.txt

# mpiexec -n $i ./mpi >>  lab2/second_fast/fast.txt

i=$(( i + 1 ))
done
