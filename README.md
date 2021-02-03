# Vigilant Driving 
Deep Learning on the road.
### Requirements

- Dataset needed can be found [here](https://github.com/commaai/speedchallenge/tree/master/data)
- Requires [Pytorch](https://pytorch.org/) to run.
- Requires [Captum](https://captum.ai/) for model interpretability.

# SEGNET 

## Visuals


# SPEED
## Test mode:
- Run main.py. **MUST** have a saved model ready.
```sh
$ cd vigilant-driving/speed
$ python main.py 
```
## Training Mode
- Use "-train" to set to training mode.
- Use "-epochs" to set the training epochs. Default is set to 1.
```sh
$ cd vigilant-driving/speed
$ mkdir models
$ python main.py -train True -epochs 1000
```

## Model Interpretability 
<img src="etc/actual.jpg" alt="actual" width="600"/>
<img src="etc/interpret.jpg" alt="interpet" width="600"/>

## Important notes
- For first time user please comment out the code below until a model has been saved. You cannot load a model that doesn't exist. Can be found in the train function.
```python
# model.load()
```
- In order to save the model uncomment the code below in the train function. 
```python
model.save()
```
- Downloading the dataset may be better than using links. If done, please set the directories of the videos and text file in the code.
## SCORE ACHIEVED
- MSE: 2.6 ~ 2.8

# Directory Structure
------
    .
    ├── Segnet              # Segmentation on lanes
        ├── dataset.py      #  Class to hold for the dataset
        ├── main.py         #  Main controller
        ├── network.py      #  Segmentation model
        ├── train.py        #  Train and test function
    └── Speed               #  Speed Prediction
        ├── main.py         #  Main controller
        ├──test_pred.txt    #  Prediction for the training set
        ├──train_pred.txt   #  Prediction on the testing set
		└── etc									# Holds  images, gif, random files 


------

# License
----

MIT
