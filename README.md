# Semantic segmentation with Convolutional Neural Networks

Autonomous cars require strong perception systems. One of the methods to have this strong perception is the semantic segmentation of the elements in the road, using convolutional neural networks. In this project, use is made of the [ERFNet](https://github.com/Eromera/erfnet) architecture for the convolutional neural network and the database [BDD100K](https://arxiv.org/abs/1805.04687) to produce a semantic segmentation in real time for the following labels:

1. Pedestrians
2. Road Curb
3. White lane 
4. Double yellow lane
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

![Test1](/Images/Test1.PNG)

![Test1_predict](/Images/Test_pred.PNG)

![Test2](/Images/Test2.PNG)

![Test2_predict](/Images/Test2_pred.PNG)

Finally, the best results were the following:

| Classes        | IOU%           |
| ------------- |:-------------:|
| Pedestrians      | 76.60 |
| Road Curb   | 44.31      |
| White Lane | 32.53      |
| Double yellow lane      | 63.38 |
| Yellow lane      | 60.61      |
| Small vehicles | 37.70      |
| Medium vehicles      | 74.38 |
| Big vehicles      | 93.49      |
| Drivable lane | 88.24      |
| Alternative lane | 76.27      |

The IOU value, is a relation in between True Positives, False Positives and Falses Negatives, described by the next equation:

![Equation](/Images/equation.PNG)

where:

- TP: True positives
- FP: False positives
- FN: False negatives

The real-time testing was implemented using the framework ROS, in the Kinetic version, and rosbag files. The semantic segmentation in real-time looks as the following:

![Test1](/Images/Realtime_test.PNG)

![Test2](/Images/Realtime_test2.PNG)

![Test3](/Images/Realtime_test3.PNG)

![Test4](/Images/Realtime_test4.PNG)

![Test5](/Images/Realtime_test5.PNG)

![Test6](/Images/Realtime_test6.PNG)

![Test7](/Images/Realtime_test7.PNG)

![Test8](/Images/Realtime_test8.PNG)

![Test9](/Images/Realtime_test9.PNG)

![Test10](/Images/Realtime_test10.PNG)

![Test11](/Images/Realtime_test11.PNG)

![Test12](/Images/Realtime_test12.PNG)
