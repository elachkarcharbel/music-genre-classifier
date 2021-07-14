# Combining Reduction and Dense Blocks for Music Genre Classification
This repo contains a music genre classifier built using Reduction Block B of Inception-v4 and Dense Blocks. Using the same code we were able to achieve an accuracy of 97.51% and 74.39% over the GTZAN dataset and the small-subset of the FMA dataset respectively.
Kindly note that we used some block of code from the following repository since our CNN architecture is inspired by the BBNN network.
https://github.com/CaifengLiu/music-genre-classification

## Main requirements:
  1. Python 3.8.5
  2. Tensorflow-gpu 2.2.0
  3. Keras 2.3.0 

#### To install the requirements before configuring and starting the training:
  1. cd to the directory where requirements.txt is located.
  2. activate your virtual environment.
  3. run the following command to install all the requirements
  
```bash
pip install -r requirements.txt
```

## Training
The training consist of converting music/audio tracks to spectrograms, slicing the spectrograms into normalized slices in width and height, and finally training the sices through the proposed network.

### Parameters
In the train.py, the parameters were configured in order to train the small-subset of the FMA dataset.
We enumerate below the paramters that should be changed in order to run the code on a custom dataset.

  1. Change the ``` dataset_path ``` variable based on the path to your dataset
  2. The ``` get_all_music ``` and ``` slice_spectrograms ``` are responsable to generate and slice the greyscale spectrograms. If you wish to save the slices in   other directories their corresponding parameters should be changed.
  3. Change the first parameter of the ``` create_dataset_from_slices ``` function based on the number of slices per genre in the use case.
  4. Change the ```num_classes ``` value based on the number of classes in the use case.

### Hyperparameters
Concerning the hyperparameters, they are configured already for performing 10-Fold Cross Validation over the dataset.
You can change the Fold value as well as following hyperparameters based on the use case:

  1. ``` k_fold ``` : Number of folds
  2. ``` num_classes ```: Number of classes in the dataset
  3. ``` train_size ```: The training set ratio
  4. ``` val_size ```: The validation set ratio
  5. ``` test_size ```: The testing set ratio

  6. ``` epochs ```: The epochs size
  7. ``` batch_size ```: The Batch size
  8. ``` lr ```: The initial learning rate value

At this stage, you are ready to initiate the training by running the ```train.py``` using the python command.
The results should be saved in the ```results/``` directory based on the configuration of the following parameters:

1. ```file_name0```: Path to the resulting models after each fold
2. ``` path``` : Path to the log files
3. ```csv_name0```: Path to the accuracy results at each fold  
  


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
