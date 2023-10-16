training_mode=${1}
data_mode=${2}
seed=${3}
store_path=${training_mode}
if [[ $data_mode =~ "_unans_k" ]]; then
	store_path="implicit_knowledge"
elif [[ $data_mode =~ "_unans_r" ]]; then
	store_path="implicit_response"
fi
echo $store_path
python finetune_model.py \
    --seed=${seed} \
    --num_train_epochs=15 \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=8 \
    --model_name_or_path="t5-base" \
    --training_mode=${training_mode} \
    --dataset_name="data/load_dataset.py" \
    --output_dir="models_bleu/${store_path}/seed${seed}" \
    --dataset_config_name=${data_mode}