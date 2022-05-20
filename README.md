# html-code-generation-from-images-with-deep-neural-networks
fyp2-two-code-generation-from-images

## Project Structure

```
.
├── base               - contains abstract class of model
├── compiler           - contains DSL compiler to bootstrap
│   ├── assets
│   └── classes
├── config             - contains neural network hyperparameters
├── data               - contains dataset and scripts to prepare data
│   ├── all_data
│   ├── eval
│   ├── img
│   │   ├── eval_images
│   │   ├── test_images
│   │   └── train_images
│   ├── test
│   └── train
├── data_loader        - data generator class inherits from Kera's Sequence
├── demo               - files for quick demo of code generation
│   └── data
│       ├── demo_data
│       └── demo_images
├── evaluator          - evaluation of model based on BLEU scores
├── generator          - code generator to generate DSL and HTML code
├── model              - contains implementation of model architecture
├── results            - contains model files & results of model training
├── trainer            - trainer used to train and fit model
└── utils              - helper functions used for callbacks & tokenizer

```

## Data Folder
```cd data``` \
```cat all_data.zip.* > all_data.zip``` \
```unzip all_data.zip```


```python split_dataset.py``` \
```python prepare_data.py```


## Trainer Folder
```python trainer.py ../results```
