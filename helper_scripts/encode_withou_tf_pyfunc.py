'''
Reference script https://colab.research.google.com/drive/1IAIeaTdl4bJlWxDJS6AXFnBCg1B92E5Z#scrollTo=NsaUPscGZ-Mq
'''

def mt_convert_examples_to_features(
                                    examples,
                                    tokenizer=tokenizer,
                                    input_max_length=512,
                                    output_max_length=72,
                                    task=None,
                                    label_list=None,
                                    output_mode=None,
                                    pad_on_left=False,
                                    pad_token=0,
                                    pad_token_segment_id=0,
                                    mask_padding_with_zero=True,
                                  ):
    
    features = []
    for (ex_index, (document, summary) ) in enumerate(examples):
        
        example = InputExample(None, document.numpy().decode('utf-8'), summary.numpy().decode('utf-8'), None)   
        input_ids = tokenizer.encode(example.text_a)
        output_ids = tokenizer.encode(example.text_b)      
        # Zero-pad up to the sequence length.
        input_padding_length = input_max_length - len(input_ids)
        output_padding_length = output_max_length - len(output_ids)
        
        input_ids = input_ids + ([pad_token] * input_padding_length)
        output_ids = output_ids + ([pad_token] * output_padding_length)

        features.append(
            InputFeatures(
                input_ids=input_ids, token_type_ids =output_ids
            )
        )

    def gen():
        for ex in features:
            yield (ex.input_ids,ex.token_type_ids)

    return tf.data.Dataset.from_generator(
        gen,
        (tf.int32,tf.int32),
        (tf.TensorShape([None]),tf.TensorShape([None]))
    )

    return features