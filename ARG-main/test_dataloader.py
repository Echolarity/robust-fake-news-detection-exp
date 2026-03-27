from utils.dataloader import get_dataloader
bert_path='/root/sunyuqi/fakenews/models/chinese-bert-wwm-ext'
try:
    dataloader=get_dataloader(
        path='data/train.json',
        max_len=128,
        batch_size=4,
        shuffle=False,
        bert_path=bert_path,
        data_type="rationale",
        language="chinese"
    )
    for batch in dataloader:
        print("=== Dataloader加载成功！Batch示例 ===")
        print(f"content_token_ids形状：{batch[0].shape}")
        print(f"label：{batch[10]}")
        print(f"source_id：{batch[11]}")
        break
except Exception as e:
    print(f"Error: {e}")
else:
    print("✅ 数据格式完全适配dataloader！")