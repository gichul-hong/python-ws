Traffic Sign Classification
U Kang
Seoul National University
U Kang 1

In This Lecture
German Traffic Sign Benchmarks
■
Multi-class classification from traffic sign images
■
U Kang 2

Outline
Introduction
Dataset
Preprocessing Codes
U Kang 3

About the Task
The German Traffic Sign Benchmark is a multi-
■
class classification challenge dataset
Link: https://benchmark.ini.rub.de/index.html
❑
Number of classes: 43
❑
Imbalanced class distribution
❑
U Kang 4

Problem Definition
Given: German Traffic Sign data
■
Classify: test data into the correct categories
■
E.g.
■
class 4
class 0
“Speed limit
“Speed limit
(70km/h)”
(20km/h)”
class 33 class 37
“Turn right “Go straight or
ahead” left”
U Kang 5

Outline
Introduction
Dataset
Preprocessing Codes
U Kang 6

Training Dataset (1)
■
U Kang 7

Training Dataset (2)
Label distribution of training data
■
U Kang 8

Test Dataset (1)
Approximately 10,000 images
■
Image sizes vary between 15x15 to 250x250
■
pixels
Images are not necessarily squared
■
U Kang 9

Test Dataset (2)
Label distribution of test data
■
U Kang 10

Outline
Introduction
Data
Preprocessing Codes
U Kang 11

Import Libraries
Import the libraries such as os, numpy, cv2, and
■
torch
U Kang 12

Loading the Dataset (1)
Extracts zip files to their corresponding folder
■
U Kang 13

Loading the Dataset (2)
Read train data and put images and labels into
■
numpy arrays
U Kang 14

Loading the Dataset (3)
Shuffle the train data
■
U Kang 15

Loading the Dataset (4)
Converts the data type to float32 and scales the
■
pixel values from 0 to 1
Splits the data into train and validation sets at an
■
8:2 ratio
U Kang 16

Test
1
2
3
U Kang 17

Test
Save results to csv file.
■
Save path: ‘./result.csv’
❑
1st column: test file name
❑
2nd column: predicted class
❑
■ Separator: comma
U Kang 18

Questions?
U Kang 19