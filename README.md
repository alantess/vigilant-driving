# Vigilant Driving 
Predicts the speed of the car from a video. 
### Requirements

- Dataset needed can be found [here](https://github.com/commaai/speedchallenge/tree/master/data)
- Requires [Pytorch](https://pytorch.org/) to run.

# Run 
## Test mode:
- Run main.py. **MUST** have a saved model ready.
```sh
$ cd vigilant-driving
$ python main.py 
```
## Training Mode
- Use "-train" to set to training mode.
- Use "-epochs" to set the training epochs. Default is set to 1.
```sh
$ cd vigilant-driving
$ python main.py -train True -epochs 1000
```

## Important notes
- For first time user please comment out the code below until a model has been saved. You cannot load a model that doesn't exist. Can be found in the train function.
```python
# model.load()
```
- In order to save the model uncomment the code below in the train function. 
```python
model.save()
```
## SCORE ACHIEVED
- MSE: 2.6 ~ 2.8

# License
----

MIT
