
model_name=NeAR
root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
random_seed=2024
k=3

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_96' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 321 \
  --n_vars 321 \
  --d_model 128 \
  --batch_size 16 \
  --d_ff 128 \
  --d_core 128 \
  --learning_rate 0.001 \
  --topK $k \
  --cycle 168 \
  --des 'Exp' \
  --freq h \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_192' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
   --e_layers 1 \
  --enc_in 321 \
  --n_vars 321 \
  --d_model 128 \
  --batch_size 16 \
  --d_ff 128 \
  --d_core 128 \
  --learning_rate 0.001 \
  --topK $k \
  --cycle 168 \
  --des 'Exp' \
  --freq h \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_336' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
   --e_layers 1 \
  --enc_in 321 \
  --n_vars 321 \
  --d_model 128 \
  --batch_size 16 \
  --d_ff 128 \
  --d_core 128 \
  --learning_rate 0.001 \
  --topK $k \
  --cycle 168 \
  --des 'Exp' \
  --freq h \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_720' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720\
   --e_layers 1 \
  --enc_in 321 \
  --n_vars 321 \
  --d_model 128 \
  --batch_size 16 \
  --d_ff 128 \
  --d_core 128 \
  --learning_rate 0.001 \
  --topK $k \
  --cycle 168 \
  --des 'Exp' \
  --freq h \
  --itr 1