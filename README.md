# Semantic segmentation with Convolutional Neural Networks

Autonomous cars require strong perception systems. One of the methods to have this strong perception is the semantic segmentation of the elements in the road, using convolutional neural networks. In this project, use is made of the [ERFNet](https://github.com/Eromera/erfnet) architecture for the convolutional neural network and the database [BDD100K](https://arxiv.org/abs/1805.04687) to produce a semantic segmentation in real time for the following labels:

1. Pedestrians
2. Road Curb
3. White lane 
4. Doble yellow lane
5. Yellow lane
6. Small vehicles
7. Medium vehicles
8. Big vehicles
9. Drivable lane
10. Alternative lane

The algorithms developed for the lanes and obstacles detection were implemented using Python 2.7. This implementation add a weight matrix to keep a balance in between the classes at the training process. Also, now it's possible to start the training since the last epoch who has been trained. 

When the training process ends the following files are generated:

- automatedlog.txt: This file keep the values corresponding to the learning rate and the precision values in each epoch

- best.txt: have the number of the epoch with the best learning rate

- Results in each epoch: For each epoch there are generated 2 text files who indicates the learning rate for each class in training and validation

- Best model of the network: save the model of the network who produces the best results and their parameters

Here are some of the examples of the images for the training set, after setting the required labels:

![Tux, the Linux mascot](/Images/Test1.PNG)

![Tux, the Linux mascot](/Images/Test_pred.PNG)

![Tux, the Linux mascot](/Images/Test2.PNG)

![Tux, the Linux mascot](/Images/Test2_pred.PNG)

Finally, the best results 




