# Importing Libraries
import numpy as np
import pandas as pd

# Importing dataset

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)  # delimiter='\t': This tells the read_csv function that the columns in the file are separated by a tab (\t).
                                                                          # quoting=3: A value of 3 corresponds to csv.QUOTE_NONE, which indicates that no characters should be quoted in the file. 
                                                                          # In this case, no quotation marks are used in the data.

# Cleaning texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,len(dataset)):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #This line replaces all non-letter characters 
                                                          #(numbers, punctuation, special characters, etc.)
                                                          # in each review with a space character.
    
    review = review.lower() # This line converts all letters in the review to lower case.

    review = review.split() #By default, it splits the text over the space character. 
                            # That is, each space in the text 

    ps = PorterStemmer()  #Creates an object of the PorterStemmer class and assigns it to the ps variable.
                          # The ps object can now be used to stem a word.

    all_stopwords = stopwords.words('english')   # function returns a list of stopwords commonly used in the English language.

    all_stopwords.remove('not') # This Line Removes “not” from the list of English stopwords.

    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]   # This line returns a list of words with stopwords removed and stemmed.

    review = ' '.join(review)   # Recombines a list of words into a single string with spaces between them.

    corpus.append(review)


# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer # CountVectorizer counts the number of times each word occurs in
                                                            #  the document and converts these frequencies into a vector.

cv = CountVectorizer(max_features=1500) # max_features=1500 parameter: This parameter ensures that the model 
                                        #  only considers the 1500 most frequent words.

X = cv.fit_transform(corpus).toarray()  # .toarray(): This method converts the resulting vectors into a NumPy array. 

y = dataset.iloc[:,-1].values

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Training the Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
bayes = GaussianNB()
bayes.fit(X_train,y_train)

# Predicting the test set results
y_pred = bayes.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(-1,1), y_test.reshape(-1,1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Confusion Matrix:",cm)
print("Accuracy Score:",accuracy)