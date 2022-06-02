from tensorflow.keras import Model, layers

def gen_lstm_model(dim_in, dim_out, optional_layers : dict):
    
    

    inputs = layers.Input(dim_in)
    LSTM = None
    outputs = layers.Dense(dim_out)
    return 
