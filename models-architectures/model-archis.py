__author__ = 'Taneem Jan, taneemishere.github.io'

# import the module from keras to load a saved model
from keras.models import load_model, model_from_json
from keras.utils.vis_utils import plot_model

# load json and create model
# json_file = open('../bin/autoencoder.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
#
# print("Loaded json model from disk")

# load weight into new model
# loaded_model.load_weights("../bin/autoencoder.h5")
# print("Loaded weights into new model")
#
# print(loaded_model.summary())
#
# plot_model(
#     loaded_model,
#     to_file='./autoencoder.png',
#     show_shapes=True,
#     show_layer_names=False,
#     show_layer_activations=True
# )
# print("Saved model to disk")

# load json and create model
json_file = open('../bin/Main_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

print("Loaded json model from disk")

# load weight into new model
loaded_model.load_weights("../bin/Main_Model.h5")
print("Loaded weights into new model")

print(loaded_model.summary())

# plot the model
plot_model(
    loaded_model,
    to_file='./main_model.png',
    show_shapes=True,
    show_layer_names=False,
    show_layer_activations=True
)
print("Saved model to disk")
