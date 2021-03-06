# Hate Speech Detection

## Progress

### Phase 1: Building a Machine Learning model to identify Hate Speech

* A dataset was downloaded from [here](https://github.com/aitor-garcia-p/hate-speech-dataset). It contains 10,944 text files, each of which is a data point. There's also a meta data file that contains classification of these data points. Additional information can be found [here](https://github.com/Vicomtech/hate-speech-dataset#repository-structure).
* Some analysis was done using rule based algorithms instead of ML models. As the dataset is skewed, it was easy to get a good accuracy, however the precision and recall values were poor.
* An SVM model was built for the same dataset. The best f1 score obtained was only around 27%, even though the accuracy was 90%. Consequently, the obtained result was less than satifactory.
