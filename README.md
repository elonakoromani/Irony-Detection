## Irony Detection in English Tweets 2017
### Installation  
#### 1. Download the project    
  
Download the project with the following command  
```git clone https://github.com/elonakoromani/Irony-Detection.git```    
  
#### 2. Install dependencies  
```pip install -r requirements.txt```   
  
Install other dependencies if needed.  
We have Preprocessed data.  
  
Files :   
feature_extraction.py - Takes a tweet as input and genearates a feature-value mapping list.  
classification_svm.py - It first generates the features for all tweets and then train a classifier usingthese features. It outputs a report showing the accuracy of the classifier.  
classifierSVM.p - SVM classifier which is trained by the training file.   
dictionaryFileSVM.p - This is the dictionary vector object to convert from the lists of feature value to vectors to train  
irony_detection.py - Takes an input from the user and gives the ironic score of the given statement.
Similar files for 3 classification algorithms are stored.  
  
#### 3. Running the Program  
  
Detect Irony :   
  
Approach One  
python classification_svm.py  // Run SVM algorithm  
In a similar way run other algorithms  
  
Approach Two  
python classification.py
  
For Ironic Score :  
python irony_detection.py //  enter the input text to console to get output score

for example :  
$ python irony_detection.py  
Enter the tweet to get the ironic score or type exit to quit   
"Oh how I love being ignored"  
87  
