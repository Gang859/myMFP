import os
import pandas as pd
pred_df =  pd.read_csv('/mnt/zhangrengang/workspace/myMFP/model_pth/submission_xgb.csv')
pred_df = pred_df.assign(
        Count=pred_df.groupby("sn_name")["sn_name"].transform("count")
    )
pred_df = pred_df[pred_df['Count'] > 5]
pred_df = pred_df.drop('Count', axis=1)

pred_df.to_csv("/mnt/zhangrengang/workspace/myMFP/model_pth/submission_xbg_extract.csv", index=None)
print(pred_df)

pred_df_temp= pred_df.copy()
pred_df_temp = pred_df_temp.drop_duplicates(subset="sn_name", keep="last")
print(pred_df_temp)