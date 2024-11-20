
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import threading
from pygtrans import Translate
import config
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

app = Flask(__name__)
model_lock = threading.Lock()
CORS(app) 

# ===================== ChromaDB ====================
chroma_client = chromadb.PersistentClient(path=config.VDB_PATH)
translate_client = Translate()

emb_fn = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=config.VDB_HUGGING_FACE_API,
    model_name=config.VDB_MODLE
)
collection = chroma_client.get_collection(name=config.VDB_COLLECTION_NAME, embedding_function=emb_fn) # 預設 VDB 已經存在
# =====================================================

def trans2target(input_sentence, target_lan):
    translated_successfully = False
    counter = 0

    # 若翻譯失敗就重新直到沒錯
    while (not translated_successfully) and counter < 150:
        try:
            counter += 1
            text = translate_client.translate(input_sentence, target=target_lan)
            if text:
                translated_successfully = True
                return text.translatedText
            else:
                print("翻譯失敗，重試中...")
        except Exception as e:
            print(f"翻譯發生錯誤: {str(e)}，重試中...")

    print('失敗太多次，回傳字串 : error')
    return 'error'


def getPaperData(query_texts):
    # 轉英文
    n_result = 1 # solve bug in chromadb
    # query_texts_eng = trans2target(query_texts, 'en')
    result = collection.query(query_texts=[query_texts], n_results=n_result)
    print('\nresult : ', result)
    if(result['distances'][0][n_result-1] > 0.5): # 太大太小都有鬼...
        print("無相關資料，距離 : ", result['distances'][0][n_result-1])
        return '', ''
    else:
        print('距離 : ', str(result['distances'][0][n_result-1]))

    result_eng = result['documents'][0][n_result-1]
    result_eng = result_eng.replace('\\n', '')
    result_ch = trans2target(result_eng, 'zh-TW') # 轉中文
    result_ch = result_ch.replace('&#39', '')

    print("英文版 : ", result['documents'][0][n_result-1])
    print("中文版 : ", result_ch)

    return result_ch, result_eng


@app.route('/myPaperVDB', methods=['POST'])
def predict_fn():
    insertValue = request.get_json()
    userQues = insertValue['question']

    # 模型回復
    with model_lock:
        print("問題:", userQues)
        paper_paragraph_ch, paper_paragraph_eng = getPaperData(userQues)
    return jsonify({'paperDataEng ' : str(paper_paragraph_eng),
                    'paperDataCh': str(paper_paragraph_ch)
                    }), 200

if  __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)