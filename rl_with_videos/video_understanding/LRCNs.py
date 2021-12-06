from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf

def LRCNs(
        input_shapes,
        output_size,
        feature_extractor = None,
        hidden_state_num = 2,
        hidden_state_size = (64, 16),
        *args,
        **kwargs):
    video = keras.layers.Input(shape=input_shapes,name='video_input')
    if feature_extractor is not None:
        feature_extractor.trainable = True
        encoded_frame = keras.layers.TimeDistributed(keras.layers.Lambda(lambda x: feature_extractor(x)))(video)
    else:
        encoded_frame = keras.layers.TimeDistributed(keras.layers.Lambda(lambda x: x))(video)
    
    for i in range(0, hidden_state_num - 1):
        encoded_frame = keras.layers.LSTM(hidden_state_size[i], return_sequences=True)(encoded_frame)
        
    encoded_vid = keras.layers.LSTM(hidden_state_size[hidden_state_num-1], return_sequences=False)(encoded_frame)
    encoded_vid = keras.layers.Dropout(0.3)(encoded_vid)
        
    # encoded_vid = keras.layers.Dense(8, activation='relu')(encoded_vid)
    outputs = keras.layers.Dense(output_size, activation='softmax')(encoded_vid)
    
    model = keras.models.Model(inputs=[video],outputs=outputs)
    
    return model