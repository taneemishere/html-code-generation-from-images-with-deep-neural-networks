# html-code-generation-from-images-with-deep-neural-networks

This automated AI system generates the HTML code right directly from uploading a UI/MockUp image to the system. There are two parts here, the encoder part captures and features the images and encode it into inner codes and features and learn the UIs, the decoder part learn the coded bootstrap code provided for that image and learn the stream while learning or training, and then concatenate the both the inner features from encoder and decoder. Then at the last the LSTMs takes those inner features and generates the intermediate bootstrap code as a result, which is then compiled into HTML code through a web compiler. The results achieved here, are higher than that of the expectations even at the evaluation set.

## Project Structure

```
.
|── bin                     - contains the model pretrained weights in .h5 and .json 
├── compiler                - contains DSL compiler to bootstrap from intermediate code of .gui format
│   ├── assets      
│   └── classes
├── datasets                - contains dataset in zip files which is linked
│   ├── all_data
│   ├── eval
│   ├── img
│   │   ├── eval_images
│   │   ├── test_images
│   │   └── train_images
│   ├── test
│   └── train
├── evaluate                - evaluation of model based on BLEU scores
├── generated-outputs       - contains the ouputs generated by the model
├── logs                    - logs of models
├── model                   - contains implementation of model architecture, the autoencoder and main model
|   ├── classes
|   |   ├── dataset           - contains datasets generators
|   └── └── model             - contains the implementation of models
└── model-architectures     - contains viualization of model architecture and thier summaries 

```

## Usage

 
> Prepare the Dataset

```
# unzip the data
cd datasets
zip -F pix2code_datasets.zip --out datasets.zip
unzip datasets.zip

cd ../model

# split training set and evaluation set 
# usage: build_datasets.py <input path> 
./build_datasets.py ../datasets/web/all_data

# transform images into numpy arrays in training set (normalized pixel values and resized pictures to smaller files if you need to upload the set to train your model in the cloud)
# usage: convert_imgs_to_arrays.py <input path> <output path>
./convert_imgs_to_arrays.py ../datasets/web/training_set ../datasets/web/training_features
```

> Model Training

```
cd model

# provide input path to training data and output path to save trained model and metadata
# usage: train.py <input path> <output path> <train_autoencoder>
./train.py ../datasets/web/training_set ../bin

# train on images pre-processed as converted to numpy arrays
./train.py ../datasets/web/training_features ../bin

# train with autoencoder
./train.py ../datasets/web/training_features ../bin 1
```

> Generate Code for an Image

```
mkdir generated-output
cd model

# generate DSL code aka the .gui file, the default search method is greedy
# usage: sample.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./sample.py ../bin pix2code2 ../test_gui.png ../generated-output

# equivalent to command above
./sample.py ../bin pix2code2 ../test_gui.png ../code greedy

# generation with beam search is coming soon
```

> Compile the .gui code to HTML

```
cd compiler

# compile .gui file to HTML/CSS (Bootstrap style)
# usage: web-compiler.py <input file path>.gui
./web-compiler.py ../generated-output/dot_gui.file
```
