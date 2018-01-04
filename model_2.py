import gensim
from gensim.models.doc2vec import TaggedDocument
from os import listdir
from os.path import isfile, join
from nltk import RegexpTokenizer
import re
# from random import shuffle

#Grab data aka documents
docLabels=[]
docLabels = [f for f in listdir("Data/Total_Data") if f.endswith('.txt')]
data=[]
for doc in docLabels:
    data.append(open('Data/Total_Data/' + doc).read())

#added
#tokenize and tag documents
tokenizer = RegexpTokenizer(r'\w+')
texts=[]
taggeddoc=[]
for ind,i in enumerate(data):
    wordslist=[]
    tagslist=[]
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    #tokens2 = ' '.join(tokens).split()
    td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(tokens))).split(),str(ind))
    taggeddoc.append(td)

documents=taggeddoc

documents=taggeddoc

model.build_vocab(documents)

#model build
model = gensim.models.Doc2Vec(documents, dm=1, alpha=0.025, min_alpha=0.025, min_count=0)


#training
for epoch in range(100):
    if epoch % 10 == 0:
        print("Training epoch {}").format(epoch)
    model.train(documents)
    model.alpha -= 0.002
    model.min_alpha = model.alpha

#show similar documents to doc 2
print(model.docvecs.most_similar(str(2)))

#similarity/distance between the documetn with ID2 and the rest
model.docvecs.most_similar(str(2))

#save model
model.save(“doc2vec.model”)





# data2=[]
# for string in data:
#     string = string.lower()
#     data2.append(string)

# random_data = data
# random.shuffle(random_data)
# train_data =  random_data[:2657]
# test_data = random_data[2657:]
#
# train_data_words = [words for segments in train_data for words in segments.split()]
# fdist = nltk.FreqDist(train_data_words)
#
# for word, frequency in fdist.most_common(50):
#     print(u'{};{}'.format(word, frequency))

class LabeledLineSentence(object):

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for uid, line in enumerate(open(filename)):
            yield LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])
