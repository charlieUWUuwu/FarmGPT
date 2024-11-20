from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
# ===================
import torch
# ===================
from pygtrans import Translate
import re
# ===================
import gc
import threading
# ===================
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
# ===================
import config

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
# 預設 VDB 已經存在
collection = chroma_client.get_collection(name=config.VDB_COLLECTION_NAME, embedding_function=emb_fn)
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available()) 
print(torch.__version__)
print(torch.version.cuda)

def load_model(model_name):
    global device

    model = None
    tokenizer = None
    if(model_name == 'gpt3-damo-base-zh' or model_name == 'gpt3-damo-large-zh') : 
        from gpt3 import GPT3ForCausalLM
        from transformers import BertTokenizerFast
        model = GPT3ForCausalLM.from_pretrained(config.BASE_GPT_MODEL_PATH)
        tokenizer = BertTokenizerFast.from_pretrained(config.GPT_TOKENIZER_PATH)
    elif(model_name == 'Bloom-1b1-zh') :
        from transformers import BloomForCausalLM
        from transformers import BloomTokenizerFast
        tokenizer = BloomTokenizerFast.from_pretrained(config.BLOOM_TOKENIZER_PATH)
        model = BloomForCausalLM.from_pretrained(config.BLOOM_MODEL_PATH)

    model = model.to(device)
    return model, tokenizer
    
def getTokenizedLength(texts, tokenizer):
    result_len = len(tokenizer(texts, add_special_tokens=False)[0].ids)
    return result_len

def addSuffix(isGPT, input_ids):
    # "答" ":"  (gpt3)   # " " "答" ":"  (bloom)
    suffix = torch.tensor([11806, 40]) if (isGPT) else torch.tensor([210, 10891, 29])
    input_ids = torch.cat((input_ids, suffix)).to(device)
    return input_ids

def getGPTPredict(model, tokenizer, input_sentence, pad_token_id):
    input_sentence = translate_client.translate(input_sentence, target='zh-CN').translatedText # 轉簡體
    input_ids = tokenizer(input_sentence, return_tensors="pt")["input_ids"][0] 
    input_ids = addSuffix(True, input_ids)

    with torch.no_grad():
        #output = model.generate(input_ids, do_sample=True, max_length=400, top_k=35) 
        output = model.generate(input_ids.unsqueeze(0), do_sample=False, max_length=768, num_return_sequences=1, repetition_penalty = 1.2, pad_token_id=pad_token_id)

        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        output_text = output_text.partition("答 : ")[2] # 獲得輸出內容

        # 去除中文字間空格，保留英文空格
        regex= re.compile(r'(?<=[^a-zA-Z0-9\u0021-\u002E])(\x20)(?=[^a-zA-Z0-9\u0021-\u002E])')
        output_text = re.sub(regex, '', output_text)
        regex= re.compile(r'(\x20)(?=[\(\%\uFF00-\uFFFF])')
        output_text = re.sub(regex, '', output_text)
        output_text = output_text.replace(' . ','.')

        output_text = translate_client.translate(output_text, target='zh-TW').translatedText # 轉繁體
        del regex, output, tokenizer
        return output_text

def getBloomPredict(model, tokenizer, input_sentence, pad_token_id):
    input_ids = tokenizer(input_sentence, return_tensors="pt")["input_ids"][0] 
    input_ids = addSuffix(False, input_ids)

    with torch.no_grad():
        print("input_ids 長度 : ", len(input_ids))
        output = model.generate(input_ids=input_ids.unsqueeze(0), do_sample=False, max_length=1024, num_return_sequences=1, repetition_penalty = 1.2, pad_token_id=pad_token_id)
        output_text = tokenizer.decode(output[0], skip_special_tokens=False) # tokenizer.decode() 的輸出"不"會在每個字中間自動加入空白格

        # 使用 split() 方法將文本按空格分割成單詞列表
        index1 = output_text.find("答:") + 2 # 不要取到"答:"
        index2 = output_text.find("</s>")
        output_text = output_text[index1:index2] # 【答:...】 ，不會包含 </s>
        print('\n模型完整輸出 : ', output_text)
        del output, index1, index2
        return output_text

def get_model_predict(input_sentence, modelName, paperDataCh):
    global device
    model, tokenizer = load_model(modelName)
    if(model == None):
        return '模型載入錯誤'
    model.eval()
    
    # 限制論文段落字數
    max_length = 250
    print("paperDataCh : ", paperDataCh)
    print('論文長度 : ', len(paperDataCh))
    if(len(paperDataCh) != 0):
        print('論文 : ', paperDataCh)
        if(len(paperDataCh) > max_length):
            paperDataCh = paperDataCh[:max_length-1] + '。'
        elif(paperDataCh[-1] != '。'):
            paperDataCh = paperDataCh[:-1] + '。'

    input_sentence = "<s>問:" + paperDataCh + "。問: " + input_sentence

    print("\n餵給模型: ", input_sentence)

    if (modelName == 'Bloom-1b1-zh') :
        output_text = getBloomPredict(model, tokenizer, input_sentence, 3)
    elif(modelName == 'pt3-damo-large-zh' or modelName == 'pt3-damo-base-zh'):
        output_text = getGPTPredict(model, tokenizer, input_sentence, 0)
    else:
        output_text = '請求走錯路到bloom了'

    output_text = output_text.replace('/', '')
    del model, input_sentence, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return output_text

# 問答模型首頁
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_fn():
    insertValue = request.get_json()
    userQues = insertValue['question']
    modelName = insertValue['modelName']
    paperDataCh = insertValue['paperDataCh']

    # 模型回復
    with model_lock:
        print("問題:", userQues)
        print("使用模型:", modelName)
        userQues = userQues.replace("碳權", "碳信用") # 替換單字
        modelAns = get_model_predict(userQues, modelName, paperDataCh)
        modelAns = modelAns.replace("碳信用", "碳權")
        print("模型輸出 : ", modelAns)
        del userQues, modelName, insertValue
        gc.collect()
        torch.cuda.empty_cache()
    return jsonify({'answer' : str(modelAns)}), 200


if  __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)

    # 以waitress生產一個WISG服務器，並以waitress.server取代 “flask run”
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5500)