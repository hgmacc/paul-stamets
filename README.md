# paul-stamets
Using ML to build my own personal Mycologist

## DATA DOWNLOAD    
- Go to: https://www.kaggle.com/maysee/mushrooms-classification-common-genuss-images
- Down kaggle images and place *unzipped* folder in `project/data`
- run `cd project/src`
- run `python3 data_preproc.py` (this will resize, grayscale and throw images into a .mat)
- `all_shrooms.mat` should exist in `project/outputs`

## Models include
- KNN Analysis
- Random forest
- Simple CNN Architecture
- SVM (Linear and poly)
- Over-arching K-Fold analysis completed on the above models
- Data analysis: AUC Heatmap performed on image data, AUC values in tabular form for other Mushroom data. 
