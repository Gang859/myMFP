train_start_month="01"
train_end_month="05"
test_end_month="06"
output_file="/mnt/zhangrengang/workspace/myMFP/output/submission_2.csv"

# full dataset
python /mnt/zhangrengang/workspace/myMFP/src/baseline.py \
  --data_path /home/zhangrengang/Doc/competition_data/stage1_feather \
  --feature_path /home/zhangrengang/Doc/release_features_${train_start_month}-${test_end_month}/combined_sn_feature_ori \
  --train_data_path /home/zhangrengang/Doc/release_features_${train_start_month}-${test_end_month}/train_data \
  --test_data_path /home/zhangrengang/Doc/release_features_${train_start_month}-${test_end_month}/test_data \
  --ticket_path /home/zhangrengang/Doc/competition_data/stage1_feather/ticket.csv \
  --output_file  ${output_file} \
  --s1_train_start_month ${train_start_month} \
  --s1_train_end_month ${train_end_month} \
  --s1_test_end_month ${test_end_month} \
  > /mnt/zhangrengang/workspace/myMFP/run_log/baseline_2.log

python /mnt/zhangrengang/workspace/myMFP/src/evaluator.py \
  --submission_file ${output_file} \
  --cutoff_date ${train_end_month} \
  > /mnt/zhangrengang/workspace/myMFP/run_log/eval_2.log