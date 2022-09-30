# The checkpoint paths of the filter model to load
model_paths[0]=

# Data paths used for testing
dev_paths[0]=


limit=0.01
###############################################################################################################################################################################################################################################################################################################################################
i_length=${#dev_paths[*]}
for ((i=0; i<i_length; i++))
do
    j_length=${#model_paths[*]}
    for ((j=0; j<j_length; j++))
    do
        echo "***********************************************************************************"
        echo ${dev_paths[i]}
        echo ${model_paths[j]}
        echo ${limit}
        python evaluate.py \
            --model_path ${model_paths[j]} \
            --test_path ${dev_paths[i]}  \
            --batch_size 16 \
            --max_seq_len 512 \
            --limit ${limit}
    done
done
