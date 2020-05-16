''' 
Borrowed from https://huggingface.co/blog/how-to-train
'''

from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
en_tokenizer = ByteLevelBPETokenizer()
ta_tokenizer = ByteLevelBPETokenizer()
'''
en_path1 = [str(x) for x in Path("D:/Local_run/Hugging_face_tokenizers_input/").glob("**/*_en.*")]
en_path2 = [str(x) for x in Path("D:/Local_run/Hugging_face_tokenizers_input/").glob("**/*.en")]
en_path3 = [str(x) for x in Path("D:/Local_run/Hugging_face_tokenizers_input/").glob("**/*.en_??")]
en_paths = en_path1+en_path2+en_path3
ta_path1 = [str(x) for x in Path("D:/Local_run/Hugging_face_tokenizers_input/").glob("**/*_ta.*")]
ta_path2 = [str(x) for x in Path("D:/Local_run/Hugging_face_tokenizers_input/").glob("**/*.ta*")]
ta_paths = ta_path1+ta_path2
'''
en_paths = 'Path to the files containing english documents'
ta_paths =  'Path to the files containing tamil documents'
en_tokenizer_path = 'en_tokenizer'
ta_tokenizer_path = 'ta_tokenizer'
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
ta_tokenizer.train(files=ta_paths, vocab_size=8300, min_frequency=2, special_tokens=[
    "<CLS>",
    "<pad>",
    "<SEP>",
    "<UNK>",
    "<MASK>",
])
en_tokenizer.save(".", f"{en_tokenizer_path}")
ta_tokenizer.save(".", f"{ta_tokenizer_path}")
tamil_text = 'அதனை நிரூபிப்பதுபோல் இருக்குமாம் படம்'
english_text = 'This movie will prove that'
id_1 = ta_tokenizer.encode(tamil_text)
assert (ta_tokenizer.decode(id_1.ids)==tamil_text), 'mismatch in tamil tokenizer encoding'
id_2 = en_tokenizer.encode(english_text)
assert (en_tokenizer.decode(id_2.ids)==english_text), 'mismatch in english tokenizer encoding'

en_tokenizer = ByteLevelBPETokenizer(f'.\\{en_tokenizer_path}-vocab.json', f'.\\{en_tokenizer_path}-merges.txt')
ta_tokenizer = ByteLevelBPETokenizer(f'.\\{ta_tokenizer_path}-vocab.json', f'.\\{ta_tokenizer_path}-merges.txt')
