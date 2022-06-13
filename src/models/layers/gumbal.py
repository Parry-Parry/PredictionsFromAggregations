from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import objectives
from tensorflow.keras.activations import softmax
from tensorflow.keras.objectives import binary_crossentropy as bce


def gumbel_loss(x, x_hat): # TODO: Expand to full custom model
    q_y = K.reshape(logits_y, (-1, N, M))
    q_y = softmax(q_y)
    log_q_y = K.log(q_y + 1e-20)
    kl_tmp = q_y * (log_q_y - K.log(1.0/M))
    KL = K.sum(kl_tmp, axis=(1, 2))
    elbo = dim * bce(x, x_hat) - KL 
    return elbo