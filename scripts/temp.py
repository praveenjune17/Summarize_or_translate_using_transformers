def mask_and_calculate_loss(mask, loss):
    mask   = tf.cast(mask, dtype=loss.dtype)
    loss *= loss * mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def loss_function(target_ids, draft_predictions, refine_predictions, model):
    # pred shape == real shape = (batch_size, tar_seq_len, target_vocab_size)
    true_ids_3D = label_smoothing(tf.one_hot(target_ids, depth=config.target_vocab_size))
    loss_object = tf.keras.losses.CategoricalCrossentropy(
                                                      from_logits=True, 
                                                      reduction='none'
                                                      )
    draft_loss  = loss_object(true_ids_3D[:, 1:, :], draft_predictions)
    draft_mask = tf.math.logical_not(tf.math.equal(target_ids[:, 1:], config.PAD_ID))
    draft_loss = mask_and_calculate_loss(draft_mask, draft_loss)
    if refine_predictions:
        refine_loss  = loss_object(target_ids_3D[:, :-1, :], refine_predictions)
        refine_mask = tf.math.logical_not(tf.math.logical_or(tf.math.equal(
                                                                target_ids[:, :-1], 
                                                                config.target_CLS_ID
                                                                          ), 
                                                             tf.math.equal(
                                                                target_ids[:, :-1], 
                                                                config.PAD_ID
                                                                          )
                                                             )
                                          )
        refine_loss = mask_and_calculate_loss(refine_mask, refine_loss)
    else:
        refine_loss = [0.0]
    regularization_loss = tf.add_n(model.losses)
    total_loss = tf.reduce_sum(draft_loss, 
                               refine_loss, 
                               regularization_loss
                              )
    return total_loss
