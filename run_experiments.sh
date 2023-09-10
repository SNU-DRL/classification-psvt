# Model
model=ecg_resnet34_psvt

# Hyperparameters
bs=32
lr=1e-4
wd=1e-4
epochs=15

# Dataset
num_folds=10
cv_folds_path='./cv_folds'
num_leads=12

# Other settings
result_path='./cv_results'
gpu_num=0

for i in `seq 0 $(($num_folds-1))`
do
    python main.py \
        --model $model \
        --batch_size $bs \
        --lr $lr \
        --epochs $epochs \
        --weight_decay $wd \
        --cv_folds_path $cv_folds_path \
        --valid_fold_num $i \
        --num_leads $num_leads \
        --result_path $result_path \
        --random_seed $i \
        --gpu_num $gpu_num
done

python analyze_all_fold_results.py \
    --result_path $result_path