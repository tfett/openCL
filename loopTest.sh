make
nvcc cudamm.cu -o cudamm
gcc cpumm.c -o cpumm

clear
echo "Type    Run     Ah      Aw      Bh      Bw      Ch      Cw      gAlloc          gCopy           gExex           initM           cAlloc          tDevice         total   "




for c in 32 64 128 256 512 1024 2048 4096 5680
do
for d in 1 2 3 4 5
do
printf "openCL\t%d\t" $d
./openclmm $c $c $c $c
done
done


for c in 32 64 128 256 512 1024 2048 4096 5680
do
for d in 1 2 3 4 5
do
printf "cuda\t%d\t" $d
./cudamm $c $c $c $c
done
done

for c in 32 64 128 256 512 1024 2048 4096 5680
do
for d in 1 2 3 4 5
do
printf "cpu\t%d\t" $d
./cpumm $c $c $c $c
done
done
