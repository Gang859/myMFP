train_start_month="01"
train_end_month="05"
test_end_month="06"

# full dataset
python /mnt/zhangrengang/workspace/myMFP/src/bsl_zzx.py \
  --data_path /home/zhangrengang/Doc/competition_data/stage1_feather \
  --feature_path /home/zhangrengang/Doc/release_features_${train_start_month}-${test_end_month}/combined_sn_feature_zzx_1 \
  --train_data_path /home/zhangrengang/Doc/release_features_${train_start_month}-${test_end_month}/train_data \
  --test_data_path /home/zhangrengang/Doc/release_features_${train_start_month}-${test_end_month}/test_data \
  --ticket_path /home/zhangrengang/Doc/competition_data/stage1_feather/ticket.csv \
  --output_file /mnt/zhangrengang/workspace/myMFP/output/submission_1.csv \
  > /mnt/zhangrengang/workspace/myMFP/run_log/baseline.log

# sample dataset
# python /mnt/zhangrengang/workspace/myMFP/src/baseline.py \
#   --data_path /mnt/zhangrengang/workspace/myMFP/sample_data \
#   --feature_path /mnt/zhangrengang/workspace/myMFP/sample_data/release_features_${train_start_month}-${test_end_month}/combined_sn_feature \
#   --train_data_path /mnt/zhangrengang/workspace/myMFP/sample_data/train_data \
#   --test_data_path /mnt/zhangrengang/workspace/myMFP/sample_data/test_data \
#   --ticket_path /mnt/zhangrengang/workspace/myMFP/sample_data/ticket.csv \
#   --output_file /mnt/zhangrengang/workspace/myMFP/output/submission_1.csv \
#   > /mnt/zhangrengang/workspace/myMFP/run_log/baseline.log