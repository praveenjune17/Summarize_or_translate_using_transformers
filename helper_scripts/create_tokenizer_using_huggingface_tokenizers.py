from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
en_tokenizer = ByteLevelBPETokenizer()
ta_tokenizer = ByteLevelBPETokenizer()
en_path1 = [str(x) for x in Path("D:/Local_run/Hugging_face_tokenizers_input/").glob("**/*_en.*")]
en_path2 = [str(x) for x in Path("D:/Local_run/Hugging_face_tokenizers_input/").glob("**/*.en")]
en_path3 = [str(x) for x in Path("D:/Local_run/Hugging_face_tokenizers_input/").glob("**/*.en_??")]
en_paths = en_path1+en_path2+en_path3
ta_path1 = [str(x) for x in Path("D:/Local_run/Hugging_face_tokenizers_input/").glob("**/*_ta.*")]
ta_path2 = [str(x) for x in Path("D:/Local_run/Hugging_face_tokenizers_input/").glob("**/*.ta*")]
ta_paths = ta_path1+ta_path2
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
en_tokenizer.save(".", "en_tokenizer")
ta_tokenizer.save(".", "ta_tokenizer")

en_tokenizer = ByteLevelBPETokenizer('.\\en_tokenizer-vocab.json', '.\\en_tokenizer-merges.txt')
ta_tokenizer = ByteLevelBPETokenizer('.\\ta_tokenizer-vocab.json', '.\\ta_tokenizer-merges.txt')

tamil_text = 'அதனை நிரூபிப்பதுபோல் இருக்குமாம் படம்'
english_text = 'This movie will prove that'
id_1 = ta_tokenizer.encode(tamil_text)
assert (ta_tokenizer.decode(id_1.ids)==tamil_text), 'mismatch in tamil tokenizer encoding'
id_2 = en_tokenizer.encode(english_text)
assert (en_tokenizer.decode(id_2.ids)==english_text), 'mismatch in english tokenizer encoding'