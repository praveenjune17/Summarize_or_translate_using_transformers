Deep learning Framework :- Tensorflow 2  

### Project features   
a)Implemented three transformer architectures that can perform Machine translation or Text summarization.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)[Transformer](https://www.tensorflow.org/tutorials/text/transformer#create_the_transformer)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)[Transformer with pointer generator](https://arxiv.org/pdf/1902.09243v2.pdf)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)[Text generation using BERT](https://arxiv.org/pdf/1902.09243v2.pdf)   
b)Beam-search and topk_topp filtering to generate text ,[Refer](https://huggingface.co/blog/how-to-generate)  
c)Used Huggingface's Transformers library for tokenization and to extract BERT embeddings  
d)Mixed precision policy enabled training  
e)BERT score, BLEU and ROUGE for evaluation    
f)sanity_check.py script to test the components of the architecture  
h)Use the trained model to visualize the embeddings of the source and target sentences in the valdiation dataset using helper_scripts/visualize_sentence_embeddings.py  
i)For translation, create bias file (when using the bertified transformer) for training by running /helper_scripts/create_bias_for_multilingual.py  ,this could help in faster convergence. Refer [init_well section in this link](http://karpathy.github.io/2019/04/25/recipe/)
## Instructions to train the model  
Change the file paths in the configuration.py and run train.py  
