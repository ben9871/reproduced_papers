IMPLEMENTATION="implementation.py"

PCA_DIMS=(16)
KERNEL_MODES=(6 8 16 18)
NB_KERNEL=(1 2 3 4)
DATASETS=("fashionmnist" "mnist")
for dataset in "${DATASETS[@]}"; do
    for pca in "${PCA_DIMS[@]}"; do
        for kmode in "${KERNEL_MODES[@]}"; do
            for kfeat in "${NB_KERNEL[@]}"; do
                if [[ ${pca} -gt ${kmode} ]]; then
                    continue
                fi
                echo "Running dataset=${dataset} PCA=${pca} kmode=${kmode} kfeat=${kfeat}"
                python3 "${IMPLEMENTATION}" \
                    --model qconv \
                    --pca_dim "${pca}" \
                    --steps 200 \
                    --seeds 3 \
                    --nb_kernels "${kfeat}" \
                    --kernel_size 2 \
                    --kernel_modes "${kmode}" \
                    --stride 1 \
                    --dataset "${dataset}"
            done
        done
    done
done
