Hannah Macdonell 
201805586

# Question 1

## a) Which classifier performs the best in this task?
The best classifier was the SVM using the rbf kernal for the 8/9 data classification. 

## b) Why do you think this classifier outperforms the others? 
This is likely due to the SVM's ability to work well with high dimentional data. The number images have 784 features to
 work with, but the SVM only focuses on a few support vectors to produce a the decision-boundary hyperplane. 

## c) How does KNN compare to the results obtained in assignment 1? Why do you observe this comparative pattern? 
The knn model scores for both image data sets were similar in their performance relative to number of neighbours
 selected. The image data has the same number of features (28x28 image) which likely contributes to the similarity in performances for knn values of 1, 5 and 10. 


# Question 2
## Describe the dataset you have collected: 
Total number of samples: 8124
Total number of measurements: 19
Classes of interest: 
       'Poisonous': 3916 - 48.2%
       'Edible': 4208 - 51.8%

Our group of interest are samples categorized as poisonous (labelled as 1s in dataset). We want to be able to properly
 classify poisonous mushrooms based on visual characteristics so our machines don't kill people. 
 
 The dataset is entirely comprised of nominal data describing the edibility of 23 different mushrooms from the
  Agaricus and Lepiota mushroom family. The recorded qualitative data is described below. 

## Dataset Description
|        Feature        |  Description                                             |
|:----------------------|:--------------------------------------------------------:|
| habitat               | Habitat the mushroom was found in i.e. forested          |
| population            | Was the mushroom alone or in a cluster?                  |
| spore-print-color     | Colour of spores i.e brown                               |
| ring-type             | Shape of mushroom ring i.e. flared                       |
| ring-number           | Either none, one or two rings on a mushroom              |
| veil-color            | Colour of the veil                                       |
| veil-type             | Type of mushroom veil                                    |
| stalk-below-ring      | Describing texture of stalk below ring                   |
| stalk-above-ring      | Describing texture of stalk below ring                   |
| stalk-shape           | Shape of stalk                                           |
| gill-color            | What is the gill colouration?                            |
| gill-size             | How large are the gills?                                 |
| gill-spacing          | How spaced are the mushroom gills?                       |
| gill-attachment       | Are the gills attached                                   |
| odor                  | Mushrooms can have almondy, spicy scents etc.            |
| bruises?              | Presence/absence                                         |
| cap-color             | Colouration of mushroom cap                              |
| cap-surface           | Texture of cap surface                                   |
| cap-shape             | Shape of mushroom cap                                    |

## AUC Values
- Also submitted in .json file -

|        Feature        |  AUC  |
|:----------------------|:-----:|
| veil-color            | 0.011 |
| cap-color             | 0.016 |
| stalk-color-abv-ring  | 0.040 |
| stalk-color-bel-ring  | 0.040 |
| cap-shape             | 0.041 |
| odor                  | 0.042 |
| population            | 0.045 |
| spore-print-color     | 0.052 |
| stalk-surf-abv-ring   | 0.120 |
| stalk-surf-bel-ring   | 0.123 |


# Question 3
##  is the best performing classifier from Question 1 the same in Question 3? 
The best performing classifier for the mushroom data set was the random forest (trees = 100) with an mean accuracy of
 100%. 
  
## Elaborate on those similarities/differences – what about your dataset may have contributed to the differences/similarities observed?
Some of RBF's strengths include accomodating missing data or working with high-dimentional data. Although neither of
 these are at paly with the mushroom data set, it still performs perfectly. 

