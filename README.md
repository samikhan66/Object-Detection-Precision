# Object Detection Precision
This repo helps to calculate the precision between a predicted boundary box and any image annotation tool. It calculates the average precision of each object per image in addition to an overall precision.

TensorFlow Object Detection API is a great and easy tool for object detection needs. However, there is no way to programmatically check accuracy/precision and manually checking test images can take up to hours.

This script eliminates the manual intervention of testing and finds the precision of generated boundary boxes. An overall 85% precision is usually considered good but your need might vary. I would still manually go through some files because occassionally, the average precision 

This script was used to check the precision on common objects such as bottle, cup, chair, bowl, laptop etc. It is currently limited to finding precision of an object occuring upto two times per image but I aim to add make it multiple some time soon.

Please free to use this to your needs but please give credit to my GitHub.Also, feel free to contribute, report bugs and recommend changes.

Cheers and email me on msamikhhan@gmail.com if you have any questions. 



