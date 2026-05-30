#!/bin/sh
#PBS -N ivf_hnsw
#PBS -e test_ivf_hnsw.e
#PBS -o test_ivf_hnsw.o
#PBS -l nodes=1:ppn=1

NODES_LIST=$(cat $PBS_NODEFILE | sort | uniq)

echo "======== PBS resource info ========" 1>&2
echo "  nodes=1  ppn=1  (IVF+HNSW single-thread)" 1>&2
cat $PBS_NODEFILE 1>&2
echo "===================================" 1>&2

for node in $NODES_LIST; do
    scp master_ubss1:/home/${USER}/ann/main_ivf_hnsw ${node}:/home/${USER}/ 1>&2
done

/home/${USER}/main_ivf_hnsw

rm /home/${USER}/main_ivf_hnsw
