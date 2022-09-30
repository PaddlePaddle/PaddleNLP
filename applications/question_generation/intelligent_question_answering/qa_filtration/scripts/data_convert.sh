# Source file path for converting to training or testing data
source_file_path=
# Used as the target file path of the train set or test set
target_file_path=
###############################################################################################################################################################################################################################################################################################################################################
python -u convert_to_uie_format.py \
            --source_file_path  $source_file_path \
            --target_file_path $target_file_path




# Source file paths for converting to training or testing data
# Used as the targets file paths of the train set or test set
source_file_paths[0]=
target_file_paths[0]=
source_file_paths[1]=
target_file_paths[1]=
###############################################################################################################################################################################################################################################################################################################################################
j=${#source_file_paths[*]}
for ((i=0; i<j; i++))
do
    echo ${source_file_paths[i]}
    echo ${target_file_paths[i]}
    python -u convert_to_uie_format.py \
                --source_file_path ${source_file_paths[i]} \
                --target_file_path ${target_file_paths[i]} 
done

