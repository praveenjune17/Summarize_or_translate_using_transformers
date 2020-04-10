import io
import numpy as np
import string
import time
from google.colab import files
from transformers import BertTokenizer, TFBertModel

table = str.maketrans(dict.fromkeys(string.punctuation))  # OR {key: None for key in string.punctuation}

def embedding_projector_files(pretrained_model_to_use, paragraph, agg='sum', filename=str(time.time())):
  words = []
  vecs  = []
  filename = filename+'_'+agg
  out_v = io.open(f'vecs_{filename}.tsv', 'w', encoding='utf-8')
  out_m = io.open(f'meta_{filename}.tsv', 'w', encoding='utf-8')
  config_json = io.open('config.json', 'w', encoding='utf-8')
  tokenizer = BertTokenizer.from_pretrained(pretrained_model_to_use)
  model = TFBertModel.from_pretrained(pretrained_model_to_use)
  embedding_layer = model.get_weights()[0]
  # Remove tabs, newlines and spaces from the paragraph
  for word in (' '.join(paragraph.split())).split():
    if word:
      # remove punctuation
      word = word.translate(table)
      ids = tokenizer.encode(word, add_special_tokens=False)
      # aggregation operation #sum, mean
      if agg=='sum':
        vec = np.sum(embedding_layer[ids,:], axis=0)
      elif agg=='mean':
        vec = np.mean(embedding_layer[ids,:], axis=0)
      out_m.write(word + "\n")
      words.append(word)
      out_v.write('\t'.join([str(x) for x in vec]) + "\n")
      vecs.append(vec)
  rows, cols = np.asarray(vecs).shape
  print(f'Shape of the embedding tensor created is {rows}, {cols}')
  config_json.write('''{
  "embeddings": [
    {
      "tensorName": "My tensor",
      "tensorShape": [
        '''+str(cols)+''',
        '''+str(rows)+'''
      ],
      "tensorPath": "https://raw.githubusercontent.com/praveenjune17/Summarize_and_translate/master/Visualize/embedding_projector_files/vecs_'''+str(filename)+'''.tsv",
      "metadataPath": "https://raw.githubusercontent.com/praveenjune17/Summarize_and_translate/master/Visualize/embedding_projector_files/meta_'''+str(filename)+'''.tsv"
    }
  ]
}''')
  assert len(vecs) == len(words), '# of words is not equal to # of embedding vecs '
  out_v.close()
  out_m.close()
  config_json.close()
  print('files are created and ready to download. Update the tensorpath and \
         metadatapath in the config file and host the config in the \
         github gist should look something like\
         https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/'+\
        'praveenjune17/9c9a399697256971d5d68357455750ac/raw/43673d4717b475895c5d26946b37d82e211f0330/embedding_info.json')
  files.download(f'vecs_{filename}.tsv')
  files.download(f'meta_{filename}.tsv')
  files.download('config.json')
  return (words, vecs)


paragraph = '''மூச்சுத் திவலை அல்லது சுவாசத் துளி (Respiratory droplet) என்பது பெரும்பாலும் நீரைக் கொண்டுள்ள ஒரு துகள் ஆகும். இது உற்பத்தி செய்யப்பட்ட பின்னர் விரைவாக தரையில் விழும் அளவுக்கு பெரியது. பெரும்பாலும் 5 மைக்ரோமீட்டருக்கும் அதிகமான விட்டம் கொண்டதாக வரையறுக்கப்படுகிறது. சுவாசிப்பது, பேசுவது, தும்மல், 
               இருமல் அல்லது வாந்தியெடுத்தல் போன்ற செயல்பாடுகளின் விளைவாக சுவாசத் துளி இயற்கையாகவே உற்பத்தி செய்யப்படுகிறது. அல்லது தூசுப்படலத்தை உருவாக்கும் மருத்துவ நடைமுறைகள், 
               கழிப்பறைகளை சுத்தப்படுத்துதல் அல்லது பிற வீட்டு வேலை நடவடிக்கைகள் மூலம் செயற்கையாகவும் இத்துளிகளை உருவாக்க முடியும்.சுவாச நீர்த்துளிகள் நீர்த்துளி உட்கருக்களிலிருந்து வேறுபட்டவையாகும். 
               நீர்த்துளி உட்கருக்கள் 5 மைக்ரோமீட்டரை விட சிறிய அளவு கொண்டவையாகும். அவை குறிப்பிடத்தக்க காலத்திற்கு காற்றில் தொங்கியிருக்க முடியும். இதனால் நீர்த்துளி உட்கருக்கள் காற்றுவழி நோய்களுக்கான நோய்க்காவிநோய்ப்பரப்பியாக இருக்கின்றன. சுவாசத் துளிகள் மூலம் காற்றுவழி நோய்கள் பரவுவதில்லை.
            '''

pretrained_model_to_use = 'bert-base-multilingual-cased'
words, vecs = embedding_projector_files(pretrained_model_to_use, paragraph, agg='mean')