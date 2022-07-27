import tensorflow.keras as tfk 
import tensorflow.math as tfm
import tensorflow as tf

def distance_loss(y_true, weights, interim_preds):
    cce = tfk.losses.CategoricalCrossentropy()
    n = len(weights)
    
    norms = [tf.norm(weights[i] - weights[j]) for i in range(n) for j in range(n)]
    norm_max = norms[tfm.argmax(norms)]
    norm_min = norms[tfm.argmin(norms)]

    minmax = lambda x : (x - norm_min) / (norm_max - norm_min)
    norms = tf.map_fn(minmax, elems=tf.constant(norms, dtype=tf.float32))

    weight_sum = tf.reduce_sum(norms, name="Norm of Weight Diff") 
    cce_sum = tfm.reduce_sum(tf.map_fn(lambda x : cce(y_true, x), elems=interim_preds), axis=0, name="Sum of CE over Generated Preds")

    #return cce_sum 
    return cce_sum - weight_sum

def ensemble_loss(y_true, interim_preds):
    cce = tfk.losses.CategoricalCrossentropy()

    return tfm.reduce_sum(tf.map_fn(lambda x : cce(y_true, x), elems=interim_preds), axis=0, name="Sum of CE over Generated Preds")