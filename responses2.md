Hannah Macdonell 
201805586

# Question 1

## a) Which classifier performs the best in this task?
A KNN model uses the closest surrounding neighbours of a data point to predict a class. Higher values of K means the
 KNN model is accounting for more surrounding neighbours, increasing the models reach to determine a class. As k
  increases, the model reaches further and further for samples to compare too, increasing the likelihood for
   miscalculation. Each dataset will have it's own optimal k value for the knn model. 

## b) Why do you think this classifier outperforms the others? 
The error rate at the lowest k value was 0. I think this is a semi-reliable estimate of performance. The KNN
 demonstrated good results for simple image data comparison, with a %0 error rate for k = 1 over 250 test samples. 

## c) How does KNN compare to the results obtained in assignment 1? Why do you observe this comparative pattern? 
df


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
## Is the profile of K vs. test error rate similar or quite different to the digit recognition example of Question 1? 
Very different. For one, my data set is nominal, meaning a knn model is not ideal for classification. That, and there
's quite likely a small error in my code becuase getting around %50 as an error rate for a binary classifier suggests
 your machine is just guessing. All the same, the results did indicate a slight increase in error rate with increased
  k values. 
  
## Elaborate on those similarities/differences – what about your dataset may have contributed to what you observe in this plot? 
Similarities - both graphs demonstrate an increase with knn values as you increase k. This an expected knn data trend
, however it is also common to see a parabola-type shape demonstrating an high error rate for low k-values and a high
 error rate for high k-values. Your optimal k-value is somewhere in between. 
 
 Differences - My graph is likely incorrect, probably to do with the data being fed into the training
  portion of the model. This aside, when examining the error graphs, my data set shows a high error rate for k=1
  , showing a very rugged parabola arc, which isn't seen in the image number classifier. 
  
  