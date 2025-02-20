# full dataset
python /mnt/zhangrengang/workspace/myMFP/src/baseline.py \
  --data_path /home/zhangrengang/Doc/competition_data/stage1_feather \
  --feature_path /home/zhangrengang/Doc/release_features_01-06/combined_sn_feature \
  --train_data_path /home/zhangrengang/Doc/release_features_01-06/train_data \
  --test_data_path /home/zhangrengang/Doc/release_features_01-06/test_data \
  --ticket_path /home/zhangrengang/Doc/competition_data/stage1_feather/ticket.csv \
  --output_file /mnt/zhangrengang/workspace/myMFP/output/submission_1.csv \
  > /mnt/zhangrengang/workspace/myMFP/run_log/baseline.log

# sample dataset
# python /mnt/zhangrengang/workspace/myMFP/src/baseline.py \
#   --data_path /mnt/zhangrengang/workspace/myMFP/sample_data \
#   --feature_path /mnt/zhangrengang/workspace/myMFP/sample_data/release_features_01-06/combined_sn_feature \
#   --train_data_path /mnt/zhangrengang/workspace/myMFP/sample_data/train_data \
#   --test_data_path /mnt/zhangrengang/workspace/myMFP/sample_data/test_data \
#   --ticket_path /mnt/zhangrengang/workspace/myMFP/sample_data/ticket.csv \
#   --output_file /mnt/zhangrengang/workspace/myMFP/output/submission_1.csv \
#   > /mnt/zhangrengang/workspace/myMFP/run_log/baseline.log