import pandas as pd
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words('english')
data = pd.read_csv('annotations_metadata.csv')

#train
temp = data.head(7660)
filenames = temp[temp.label == 'hate'].file_id
word_frequencies={}
for file in filenames:
    ifile = open('./all_files/'+file+'.txt')
    try:
        file_contents = ifile.read()
    except:
        continue
    for word in nltk.word_tokenize(file_contents):
        word = word.lower()
        if word not in stopwords and len(word)>2 and word.isalpha():
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    ifile.close()

#predict   
filenames = data.tail(3284).file_id
for file in filenames:
    ifile = open('./all_files/'+file+'.txt')
    try:
        file_contents = ifile.read()
    except:
        continue
    score, count = 0, 0
    for word in nltk.word_tokenize(file_contents):
        word = word.lower()
        if word not in stopwords and len(word)>2 and word.isalpha():
            count += 1
            score += word_frequencies.get(word.lower(),0)
    if count != 0:
        avg_score = score/count
    else:
        avg_score = 0
    print(file, avg_score)
    ifile.close()

#analyse
max_score = max(results.values())
accuracy, precision, recall, cutoffs = list(), list(), list(), list()
for i in range(10, 50):
    tp, tn, fp, fn = 0, 0, 0, 0
    cutoff = i/max_score
    for k,v in results.items():
        normal_score = v/max_score
        if normal_score > cutoff:
            if data.loc[k, 'label'] == 'hate':
                tp += 1
            else:
                fp += 1
        else:
            if data.loc[k, 'label'] == 'noHate':
                tn += 1
            else:
                fn += 1
    accuracy.append((tp+tn)/(tp+tn+fp+fn))
    precision.append(tp/(tp+fp))
    recall.append(tp/(tp+fn))
    cutoffs.append(cutoff)
plt.plot(cutoffs, accuracy, label = 'Accuracy')
plt.plot(cutoffs, precision, label = 'Precision')
plt.plot(cutoffs, recall, label = 'Recall')
plt.legend()
plt.show()