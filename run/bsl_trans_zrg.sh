train_start_month="01"
train_end_month="05"
test_end_month="06"

# full dataset
# python /mnt/zhangrengang/workspace/myMFP/src/bsl_transformer_zrg.py \
#   --data_path /home/zhangrengang/Doc/competition_data/stage1_feather \
#   --feature_path "" \
#   --train_data_path /home/zhangrengang/Doc/release_features_${train_start_month}-${test_end_month}/train_data \
#   --test_data_path /home/zhangrengang/Doc/release_features_${train_start_month}-${test_end_month}/test_data \
#   --ticket_path /home/zhangrengang/Doc/competition_data/stage1_feather/ticket.csv \
#   --output_file /mnt/zhangrengang/workspace/myMFP/output/submission_1.csv \
#   --s1_window_data_json /mnt/zhangrengang/data/dump/test.json \
#   > /mnt/zhangrengang/workspace/myMFP/run_log/baseline.log

# sample dataset
python /mnt/zhangrengang/workspace/myMFP/src/bsl_transformer_zrg.py \
  --data_path /mnt/zhangrengang/data/sample_data \
  --feature_path "" \
  --train_data_path "" \
  --test_data_path "" \
  --ticket_path /mnt/zhangrengang/data/sample_data/ticket.csv \
  --output_file "" \
  --s1_window_data_json "" \
  > /mnt/zhangrengang/workspace/myMFP/run_log/baseline.log