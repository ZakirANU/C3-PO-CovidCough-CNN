# Diagnose COVID-19 from cough signals
This study presents a novel approach to diagnosing COVID-19, a highly contagious respiratory disease that emerged in late 2019 and has since infected over 762 million people worldwide. The traditional detection methods for COVID-19, such as PCR and RAT tests, have been found to be resource-intensive and inefficient, prompting the need for alternative diagnostic methods.

This study proposed a machine learning method that diagnoses COVID-19 using cough sounds, which utilized data augmentation and segmentation techniques to increase the volume of data to more than three times the original size. In addition, ensemble models were employed to mitigate the impacts of a small dataset and imbalances within the dataset.

The model was tested on the crowdsourced Coswara dataset and a validation Russian dataset. It achieved an accuracy rate of 92.7% and an AUC of 98.1% on the Russian dataset, exceeding the method in the Covid-Envelope paper by 22% in terms of accuracy. On the Coswara dataset, the method achieved an accuracy rate of 72.3% and an AUC of 80.0%.


## Dependencies
- librosa
- audio2numpy
- pytorch
- numpy

## Downloading datasets
To get Russian dataset:

```bash
!git clone https://github.com/covid19-cough/dataset.git
```

To get Coswara dataset:
```bash
!git clone https://github.com/iiscleap/Coswara-Data.git
```


## Usage
1. Implement data preprocessing on dataset, such as augmentation, segmentation and feature extraction, using the existed preprocess file or creating a new one.
2. Use the models and function in `models.py` to train and test models.

To quickly run this project, run `train_test.ipynb` to train single models, run `train_test_ensembl.ipynb` to train ensembled models. Outputs will be generated in `examples` folder.


## Files

### Core libraries
- `models.py`: Core functions and classes, some important functions and classes are:
    - `CovidDataset`: A custom pytorch dataset used to load Covid data.
    - `CovidCNNModel`: A Pytorch model used to diagnose Covid.
    - `numberOfParam`: Calculate the number of parameters in a model
    - `testModel`: Test a model using the given testlaoder.
    - `train`: Train a given model with trainloader and valloader. Will reocrd the accuracy and loss during training, and save the file in out_dir.
    - `plotTrainingProcess`: Plot training process using data recorded by `train` function.
    - `testPerformance`:  Test and calculate the performance of the model, can save ROC curve and Output histogram in out_dir.
    - `cross_validation`: Regular cross validation
    - `cross_validation_test`: Nested Cross validation with the number of epoch to early stop as hyperparamer.
- `coviddata.py`: Implement preprocessing techniques on standard dataset, including resampling, normalisation, balancing, spliting, segmentation and augmentation.
- `feature.py`: 
    - `ForwardSelection`: A class to implement foward selection.
    - `combine_feature`: A helper function to concat different features.
- `utils/*.py`: Helper functions

### Data preprocessing
- `dataset/*/preprocess*/preprocess.ipynb`: Different preprocess processes.
- `dataset/Coswara-Data/extract_data.py`: Reorganise the dataset folder, move all useful data into one folder.

### Others
- `analysis.ipynb`: Used PCA and t-SNE to analyse cough signals and their features.
- `feature.ipynb`: An example of feature selection
- `train_test.ipynb`: An example of training and testing a model.
- `train_test_ensemble.ipynb`: An example of training and testing an ensembled model.
- `reimplementation/*.ipynb`: Implementation of others' methods, or modification of others' method.