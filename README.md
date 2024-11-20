# FarmGPT

## 資料說明
- `gpt3.py`: gpt3 模型架構設定
- `./train/dataset`: 存放問答模型訓練資料
- `./train/LLMs`: 存放訓練後的 LLM/tokenizer
- `./trai/train_model.ipynb`: 訓練模型相關程式碼
- `./inference/config.py`: 參數設定
- `./inference/paper_VDB_api.py`: chroma 向量資料庫服務
- `./inference/question_answering_api.py`: 模型問答服務

## 安裝套件
```
pip install -r requirements.txt
```

## 使用
1. 訓練模型詳見 `./train/train_model.ipynb`
2. 放置訓練過後的 LLM 於資料夾 `./train/LLMs/`
3. 修改 `./inference/config.py` 參數
4. 啟用 `my_paper_VDB.py`, `question_answering_api`
