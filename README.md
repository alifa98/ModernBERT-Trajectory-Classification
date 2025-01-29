# Training ModernBERT for Trajectory Classification

This repository contains the code for training ModernBERT for trajectory classification.
The code is based on the [Hugging Face ModernBERT](https://huggingface.co/docs/transformers/en/model_doc/modernbert#overview) implementation.



## Datasets
Datasets are [higher-order mobility data](https://zenodo.org/records/8076553).

## Dependencies
Install dependencies using the following command:

```bash
conda env create -f environment.yml
```


Activate the environment:

```bash
conda activate modern
```

## Preprocessing
We put all the hexagon/tessellation ids in a single file and then use the `transformers` library to train a tokenizer on the corpus. The tokenizer is then used to in training the model as a pre-trained tokenizer.

```bash
python preprocess.py
```

## Training a Tokenizer
We train a tokenizer on the hexagon/tessellation ids using the `transformers` library.

```bash
python tokenizer_trainer.py
```

## Training

Adjust the hyperparameters in the `train.py` file and then run the following command to train the model:

```bash
python train.py
```


## Evaluation
For evaluation of the model, run the following command:

```bash
python evaluate_script.py
```

## Running for your own data
Adjust the path of dataset and column names in each file then run the following commands.


## How can I train on my own trajectory data which is not in the hexagon/tessellation format?

For this you should map it to a hexagon/tessellation format.
You can convert to raw trajectory data to hexagon/tessellation format using the following repository:
[Point2Hex](https://github.com/alifa98/point2hex).


## Citation
To cite this repo:
```
@misc{Faraji2025ModenBERT,
  author = {Faraji, Ali},
  title = {Training ModernBERT for Trajectory Classification},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alifa98/ModernBERT-Trajectory-Classification}},
}
```
