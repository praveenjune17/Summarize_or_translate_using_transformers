from __future__ import absolute_import, division, print_function, unicode_literals
import io
import os
import numpy as np
import tensorflow as tf
import string
import time
from configuration import config
from create_model import source_tokenizer, target_tokenizer, Model
from local_tf_ops import check_ckpt
from tensorboard.plugins import projector

table = str.maketrans(dict.fromkeys(string.punctuation))  
log_dir='/logs/tensorboard_visulaization/'

# Instantiate the model
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

def tokenize_and_aggregate(tokens, tokenizer, agg, embedding_layer):
    # remove punctuation
    #tokens = tokens.translate(table)
    target_ids = tokenizer.encode(tokens)
    # create sentence embedding by aggregating the tokens
    if agg=='sum':
        embedding_vector = np.sum(embedding_layer[target_ids,:], axis=0)
    elif agg=='mean':
        embedding_vector = np.mean(embedding_layer[target_ids,:], axis=0)

    return embedding_vector

def save_checkpoint_create_config(checkpoint, config, embedding_type):
    
    checkpoint.save(os.path.join(log_dir, embedding_type+"_embedding.ckpt"))
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding_type+"_embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding_config.metadata_path = 'metadata_'+embedding_type+'.tsv'
    return config

def display_embedding_shape(sentence_embedding, sentences, embedding_type):
    sentence_embedding = np.asarray(sentence_embedding)
    rows, cols = sentence_embedding.shape
    print(f'Shape of the {embedding_type}_embedding tensor created is {rows}, {cols} and length of its meta data is {len(sentences)}')
    return sentence_embedding   

def embedding_projector_files(source_tokenizer, target_tokenizer, model, sentence_pair, agg='mean'):
    #words = []
    source_sentence_vector  = []
    target_sentence_vector = []
    souce_sentences = []
    target_sentences = []
    # Remove start and end token embedding
    target_embedding_layer = model.layers[1].get_weights()[0][1:-1, :]  
    source_embedding_layer = model.layers[0].get_weights()[0][1:-1, :]
    with open(os.path.join(log_dir, 'metadata_source.tsv'), "w", encoding='utf-8') as out_meta_source:
        with open(os.path.join(log_dir, 'metadata_target.tsv'), "w", encoding='utf-8') as out_meta_target:
    #out_meta_source = io.open(os.path.join(log_dir, 'metadata_source.tsv'), "w", encoding='utf-8')
    #out_meta_target = io.open(os.path.join(log_dir, 'metadata_target.tsv'), "w", encoding='utf-8')
            out_meta_source.write('source'+ "\t"+ 'target' + "\n")
            out_meta_target.write('source'+ "\t"+ 'target' + "\n")
    # Remove tabs, newlines and spaces from the paragraph
    for source, target in sentence_pair:
        source_embedding_vector=tokenize_and_aggregate(source, source_tokenizer, agg, source_embedding_layer)
        target_embedding_vector=tokenize_and_aggregate(target, target_tokenizer, agg, target_embedding_layer)
        #test the above
        out_meta_source.write(source+ "\t"+ target + "\n")
        out_meta_target.write(source+ "\t"+ target + "\n")
        souce_sentences.append(source)
        target_sentences.append(target)
        source_sentence_vector.append(source_embedding_vector)
        target_sentence_vector.append(target_embedding_vector)
    source_sentence_vector = display_embedding_shape(source_sentence_vector, souce_sentences, 'source')
    target_sentence_vector = display_embedding_shape(target_sentence_vector, target_sentences, 'target')
    checkpoint = tf.train.Checkpoint(source_embedding=tf.Variable(source_sentence_vector),
                                     target_embedding=tf.Variable(target_sentence_vector))
    config = projector.ProjectorConfig()
    config = save_checkpoint_create_config(checkpoint, config, 'source')
    config = save_checkpoint_create_config(checkpoint, config, 'target')
    projector.visualize_embeddings(log_dir, config)
    out_meta_source.close()
    out_meta_target.close()
    return (souce_sentences, target_sentences, source_sentence_vector, target_sentence_vector)

english_sentences = ['This is awesome', 
                    'Hi', 
                    '''sense of comedy''',
                    '''Associative Pages''',
                    '''After war started in Italy, since they already faced several losses, Italian dictator Mussolini was thrown out from his power and was arrested.''',
                    '''XML-IT IS USED TO TRANSFER MESSAGES''',
                    '''Indo-China,Madagascar,Algeria are those countries which are excluded from the list of countries were colonalism was overthrown peacefully.''',
                    '''Category: view''',
                    '''Several Soviet citizens and Poland, Latvia, Estonia, Lithuania, and German war prisoners were killed in the Kulak, Soviet's compulsory camps, as they supported Germany. 60% of Soviet war prisoners were died in the hands of Germans.''',
                    '''wealth given by identifying quality''',
                    '''The production of the Japan and Germany is increased by making the many people to to work for it.''',
                    '''After polland war the soviet union has taken forward the troop.''',
                    '''First Started in March 1944'''
                   ]
tamil_sentences = ['இது அருமை', 
                  'வணக்கம்',
                  '''கோமாளித்தனம்''',
                  '''தொடர்பான பக்கங்கள்''',
                  '''இத்தாலிய மண் மீது போர் துவங்கியதாலும் ஏற்கனவே பல தோல்விகளை சந்தித்ததாலும் இத்தாலிய சர்வாதிகாரி முசோலினி பதவியில் இருந்து தள்ளப் பட்டு கைது செய்யப்பட்டார்.''',
                  '''XML - வழங்கிக்கும் உலாவிக்குமான தகவற் பரிமாற்றத்தை ஒழுங்குபடுத்தப்பட்ட வடிவத்தில் கடத்த உதவுகிறது.''',
                  '''இந்தோ-சீனம் மடகாஸ்கர் இந்தோனேசியா அல்ஜீரியா தவிற மற்ற நாடுகளில் காலனீய ஒழிப்பு சமாதான மாக முடிந்தது.''',
                  '''பகுப்பு:உணர்வுகள்''',
                  '''இதைத் தவிர குலக் எனப்படுகிற சோவியத் கட்டாய பணி முகாம்கள்களில் போலந்தினர் லாட்வியர் எஸ்டோனியர் லிதுவேனியர் மற்றும் ஜெர்மன் போர் கைதிகள் ஜெர்மனிக்கு ஆதரவு கொடுத்ததாக குற்றம் சாட்டப்பட்ட சோவியத் குடிமகன்கள் கொல்லப் பட்டனர் .ஜெர்மானியர் கையில் 60% சோவியத் போர் கைதிகள் மடிந்தனர்.''',
                  '''தரங்கண்டு தந்த தனம்.''',
                  '''தங்கள் உற்பத்தியை பெருக்க ஜப்பானும் ஜெர்மனியும் பல மில்லியன் அடிமை தொழிலாளர்களை உட்படுத்தியது.''',
                  '''போலந்து ஆக்கிரமிப்பு பிறகு சோவியத் யூனியன் பால்டிய நாடுகளில் தன் துருப்புகளை முன்னேற்றியது.''',
                  '''முதலாவது மார்ச் 1944ல் தொடங்கிற்று.'''
               ]
sentence_pair = zip(english_sentences, tamil_sentences)
ck_pt_mgr = check_ckpt(config.checkpoint_path)
source, target, vec1, vec2 = embedding_projector_files(source_tokenizer, target_tokenizer, Model, sentence_pair, agg='mean')
if not os.path.exists(log_dir):
    try:
        shutil.rmtree(log_dir)
    except:
        pass
    os.makedirs(log_dir)


    
