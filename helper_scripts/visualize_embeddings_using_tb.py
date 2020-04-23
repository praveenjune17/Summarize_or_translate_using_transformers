from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
import time
from tqdm import tqdm
from configuration import config
from create_model import source_tokenizer, target_tokenizer, Model
from local_tf_ops import check_ckpt


ck_pt_mgr = check_ckpt(config.checkpoint_path)

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
(draft_predictions, draft_attention_weights, 
refine_predictions, refine_attention_weights) = Model(temp_input,
                                                    dec_padding_mask=None, 
                                                    enc_padding_mask=None, 
                                                    look_ahead_mask=None,
                                                    target_ids=temp_target, 
                                                    training=False, 
                                                    )


def embedding_projector_files(target_tokenizer, model, paragraph, agg='mean', filename=str(time.time())):
    words = []
    word_vector  = []
    filename = filename+'_'+agg
    out_m = io.open(os.path.join(log_dir, 'metadata.tsv'), "w", encoding='utf-8')
    # Remove start and end token embedding
    target_embedding_layer = Model.layers[1].get_weights()[0][1:-1, :]  
    # Remove tabs, newlines and spaces from the paragraph
    for word in (' '.join(paragraph.split())).split():
        if word:
            # remove punctuation
            word = word.translate(table)
            target_ids = target_tokenizer.encode(word)
            # aggregation operation #sum, mean
            if agg=='sum':
                vec = np.sum(target_embedding_layer[target_ids,:], axis=0)
            elif agg=='mean':
                vec = np.mean(target_embedding_layer[target_ids,:], axis=0)
            out_m.write(word + "\n")
            words.append(word)
            word_vector.append(vec)
    rows, cols = np.asarray(word_vector).shape
    print(f'Shape of the embedding tensor created is {rows}, {cols} and length of meta data is {len(words)}')
    
    checkpoint = tf.train.Checkpoint(embedding=tf.Variable(word_vector))
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
    #
    #assert len(vecs) == len(words), '# of words is not equal to # of embedding vecs '
    out_m.close()
    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)
    return (words, np.asarray(word_vector))


paragraph = '''புவி சூரியனிலிருந்து மூன்றாவதாக உள்ள கோள், விட்டம், நிறை மற்றும் அடர்த்தி கொண்டு ஒப்பிடுகையில் சூரிய மண்டலத்தில் உள்ள மிகப் பெரிய உட் கோள்களில் ஒன்று. இதனை உலகம், நீலக்கோள் எனவும் குறிப்பிடுகின்றனர். மாந்தர்கள் உட்பட பல்லாயிரக்கணக்கான உயிரினங்கள் வாழும் இடமான இந்த புவி, அண்டத்தில் உயிர்கள் இருப்பதாக அறியப்படும் ஒரே இடமாக கருதப்படுகின்றது. இந்தக் கோள் சுமார் 4.54 பில்லியன் ஆண்டுகளுக்கு முன்னர் உருவானது. மேலும் ஒரு பில்லியன் ஆண்டுகளுக்குள் அதன் மேற்பரப்பில் உயிரினங்கள் தோன்றின. அது முதல் புவியின் உயிர்க்கோளம் குறிப்பிடும் வகையில் அதன் வளிமண்டலம் மற்றும் உயிரற்ற காரணிகளை மாற்றியுள்ளது
            '''
words, vecs = embedding_projector_files(target_tokenizer, Model, paragraph, agg='mean', filename='temp')
if not os.path.exists(log_dir):
    try:
        shutil.rmtree(log_dir)
    except:
        pass
    os.makedirs(log_dir)