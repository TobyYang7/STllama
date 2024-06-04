# to fill in the following path to evaluation!
output_model=./checkpoints_urbangpt/tw2t_multi_reg-cla-gird
datapath=./data/ST_data_urbangpt/NYC_taxi_cross-region/NYC_taxi.json
st_data_path=./data/ST_data_urbangpt/NYC_taxi_cross-region/NYC_taxi_pkl.pkl
res_path=./result_test/cross-region/NYC_taxi
start_id=0
end_id=51920
num_gpus=4

python ./urbangpt/eval/run_urbangpt.py --model-name ${output_model}  --prompting_file ${datapath} --st_data_path ${st_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}