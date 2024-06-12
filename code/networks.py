import tensorflow as tf


# netowrk definition
def lstm_definition(input_shape, output_shape):
    # network definition
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=input_shape,return_sequences=True)) # -> 1 input layer, 1  intermediate layer
    model.add(tf.keras.layers.LSTM(128,return_sequences=True)) # -> another hidden 
    model.add(tf.keras.layers.LSTM(32)) # -> another hidden 
    model.add(tf.keras.layers.Dense(output_shape))  # -> output layer
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(),metrics=['mean_absolute_error','mean_absolute_percentage_error','mean_squared_error'])
    return model