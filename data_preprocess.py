import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


# 1. 設定 Feather 檔案的路徑
data_dir = 'data/測試資料_feather/'
pattern = os.path.join(data_dir, '銷售資料_*.feather')
file_list = glob.glob(pattern)

print(f"找到 {len(file_list)} 個 Feather 檔案，開始讀取...")

# 2. 讀取並合併所有 Feather 檔
# 讀取 Feather 檔案比 Excel 快非常多
dfs = [pd.read_feather(fp) for fp in file_list]
data = pd.concat(dfs, ignore_index=True)

print("資料合併完成！")
print(data.head(5).keys())

# 3. 統一欄位名稱
data = data.rename(columns={
    'ASID': 'user_id',
    'USER_ID': 'streamer_id',
    '商品ID': 'item_id',
    '商品名稱': 'item_name',
    '單價': 'unit_price',
    '數量': 'quantity',
    '總金額': 'total_amount',
    '付款方式': 'payment_method',
    '寄送地址': 'shipping_address',
    '下單日期': 'order_date',
    'POST_ID': 'post_id',
    '留言': 'comment',
    '時間戳記': 'order_time',
})


print(data.keys())

# 4. 轉換時間欄位型別
# 假設原始時間欄位叫 'timestamp'，格式如 '2022-01-15 14:23:00'
data['order_date'] = data['order_date'].astype(str).str.zfill(8)
data['datetime_str'] = data['order_date'].astype(str) + ' ' + data['order_time'].astype(str)

data['timestamp'] = pd.to_datetime(data['datetime_str'], format='%Y%m%d %H:%M:%S')
print("轉換失敗的筆數：", data['timestamp'].isna().sum())
print(data[['order_date', 'order_time', 'timestamp']].head())

# 5. 處理重複與缺失值
# 刪除完全重複的整列
data = data.drop_duplicates()
# 檢視缺失值比例
print("各欄位缺失值比例：")
print(data.isna().mean())

# 6. 基本統計
total_transactions = len(data)
unique_users       = data['user_id'].nunique()
unique_streamers   = data['streamer_id'].nunique()
# 如果有 item_id 欄位
# unique_items       = data['item_id'].nunique()

print(f"總交易筆數：{total_transactions}")
print(f"獨立用戶數：{unique_users}")
print(f"獨立直播主數：{unique_streamers}")
# print(f"獨立商品數：{unique_items}")

# 7. 時序分佈（可視化）
#  7.1 每月交易量
monthly = data.set_index('timestamp').resample('ME').size()
print("\n每月交易量：")
print(monthly)

#  7.2 每日交易量
daily = data.set_index('timestamp').resample('D').size()
print("\n每日交易量前五：")
print(daily.head())

# 若要畫圖，可再加：
monthly.plot(title='Monthly Transactions')
plt.show()

print("移除前筆數：", len(data))
data = data[data['unit_price'] != 0].reset_index(drop=True)
print("移除後筆數：", len(data))

data.to_csv('data/processed_sales_data.csv', index=False, encoding='utf-8-sig')