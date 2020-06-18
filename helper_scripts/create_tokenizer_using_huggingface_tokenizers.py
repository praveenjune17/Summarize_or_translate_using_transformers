''' 
Borrowed from https://huggingface.co/blog/how-to-train
'''

from tokenizers import ByteLevelBPETokenizer, Tokenizer
from pathlib import Path
import os
en_tokenizer = ByteLevelBPETokenizer()
ta_tokenizer = ByteLevelBPETokenizer()

file_path = '/content/drive/My Drive/Tensorflow_datasets/en_tam_parallel_text_dataset/downloads/extracted/'
en_path1 = [str(x) for x in Path(file_path).glob("**/*_en.*")]
en_path2 = [str(x) for x in Path(file_path).glob("**/*.en")]
en_path3 = [str(x) for x in Path(file_path).glob("**/*.en_??")]
en_paths = list(set(en_path1+en_path2+en_path3))
ta_path1 = [str(x) for x in Path(file_path).glob("**/*_ta.*")]
ta_path2 = [str(x) for x in Path(file_path).glob("**/*.ta*")]
ta_paths = list(set(ta_path1+ta_path2))


new_ta_path = []
for f in ta_paths:
  if not os.path.isdir(f):
    new_ta_path.append(f)
#en_paths = 'Path to the files containing english documents'
#ta_paths =  'Path to the files containing tamil documents'
en_tokenizer_path = 'en_tokenizer-vocab.json'
ta_tokenizer_path = 'ta_tokenizer-vocab.json'
# Customize training
en_tokenizer.train(files=en_paths, vocab_size=8300, min_frequency=2, special_tokens=[
    "<CLS>",
    "<pad>",
    "<SEP>",
    "<UNK>",
    "<MASK>",
])
print('en completed')
# Customize training
ta_tokenizer.train(files=new_ta_path, vocab_size=8300, min_frequency=2, special_tokens=[
    "<CLS>",
    "<pad>",
    "<SEP>",
    "<UNK>",
    "<MASK>",
])
print('ta completed')
en_tokenizer.save(en_tokenizer_path)
ta_tokenizer.save(ta_tokenizer_path)
en_tokenizer = Tokenizer.from_file(en_tokenizer_path)
ta_tokenizer = Tokenizer.from_file(ta_tokenizer_path)
tamil_text = 'அதனை நிரூபிப்பதுபோல் இருக்குமாம் படம்'
english_text = 'This movie will prove that'
id_1 = ta_tokenizer.encode(tamil_text)
assert (ta_tokenizer.decode(id_1.ids)==tamil_text), 'mismatch in tamil tokenizer encoding'
id_2 = en_tokenizer.encode(english_text)
assert (en_tokenizer.decode(id_2.ids)==english_text), 'mismatch in english tokenizer encoding'
