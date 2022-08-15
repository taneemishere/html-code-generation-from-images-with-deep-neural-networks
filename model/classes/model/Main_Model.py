__author__ = 'Taneem Jan using the auto-encoder version'

from keras.layers import Input, Dense, Dropout, RepeatVector, LSTM, concatenate, Flatten
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from keras import *
from .Config import *
from .AModel import *
from .autoencoder_image import *


class Main_Model(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "Main_Model"

        visual_input = Input(shape=input_shape)

        # Load the pre-trained autoencoder model
        autoencoder_model = autoencoder_image(input_shape, input_shape, output_path)
        autoencoder_model.load('autoencoder')
        autoencoder_model.model.load_weights('../bin/autoencoder.h5')

        hidden_layer_model_freeze = Model(inputs=autoencoder_model.model.input,
                                          outputs=autoencoder_model.model.get_layer('encoded_layer').output)
        hidden_layer_input = hidden_layer_model_freeze(visual_input)

        # Additional layers before concatenation
        hidden_layer_model = Flatten()(hidden_layer_input)
        hidden_layer_model = Dense(1024, activation='relu')(hidden_layer_model)
        hidden_layer_model = Dropout(0.3)(hidden_layer_model)
        hidden_layer_model = Dense(1024, activation='relu')(hidden_layer_model)
        hidden_layer_model = Dropout(0.3)(hidden_layer_model)
        hidden_layer_result = RepeatVector(CONTEXT_LENGTH)(hidden_layer_model)

        # Make sure the loaded hidden_layer_model_freeze will no longer be updated
        for layer in hidden_layer_model_freeze.layers:
            layer.trainable = False

        # The same language model used by Tony Beltramelli
        language_model = Sequential()
        language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(LSTM(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([hidden_layer_result, encoded_text])

        decoder = LSTM(512, return_sequences=True)(decoder)
        decoder = LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def fit_generator(self, generator, steps_per_epoch):
        self.model.summary()
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)
        self.save()

    def predict(self, image, partial_caption):
        return self.model.predict([image, partial_caption], verbose=0)[0]

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)
