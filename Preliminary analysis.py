import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk import wordnet as wn
import seaborn as sns
#import networkx as nx
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from rake_nltk import Rake
import copy
import re
from sklearn import linear_model
#from sklearn import LatentDirichletAllocation
import gensim
from gensim import corpora

#############################################################
##                                                         ##
##                                                         ##
##                                                         ##
##               DICTIONARY OF TAGS                        ##
##                                                         ##
##                                                         ##
##                                                         ##
#############################################################

file_list = []
tag_dict = {}
categories = set()
sub_categories = set()
tags = []
    
for i in range(10000):
    file_list.append("E:\\Acads\\Applied Machine Learning\\Final Exam\\Training\\tags_train\\" + str(i) + ".txt")
    file_object  = open(file_list[i],'r')
    p = []
    for j in file_object:    
        p = j.split(':')
        
        categories.add(p[0])
        a1 = p[1].replace("\n","") 
        
        sub_categories.add(a1)
        
        tags.append(p[0])
        tags.append(a1)

        tag_dict.setdefault(i, [])
        tag_dict[i].append(p[0])
        tag_dict[i].append(a1)
        
#############################################################
##                                                         ##
##                                                         ##
##                                                         ##
##               INPUT DATA                                ##
##                                                         ##
##                                                         ##
##                                                         ##
#############################################################
ROOT = "E:/Acads/Applied Machine Learning/Final Exam/Training/"
ROOT_TEST = "E:/Acads/Applied Machine Learning/Final Exam/Testing/"

TRAIN_SET = {
	"descriptions" : {
		"filename" : ["descriptions_train/{}.txt".format(i) for i in range(10000)],
		"data" : []
	},
	"features" : {
		"filename" : [
			"features_train/ffeatures_resnet1000_train.csv",
			"features_train/features_resnet1000intermediate_train.csv"
		],
		"data" : []
	},
	"images" : {
		"filename" : ["images_train/{}.jpg".format(i) for i in range(10000)],
		"data" : []
	},
	"tags" : {
		"filename" : ["tags_train/{}.txt".format(i) for i in range(10000)],
		"data" : []
	}
}

TEST_SET = {
	"descriptions" : {
		"filename" : ["descriptions_test/{}.txt".format(i) for i in range(2000)],
		"data" : []
	},
	"features" : {
		"filename" : [
			"features_test/features_resnet1000_train.csv",
			"features_test/features_resnet1000intermediate_train.csv"
		],
		"data" : []
	},
	"images" : {
		"filename" : ["images_test/{}.jpg".format(i) for i in range(2000)],
		"data" : []
	},
	"tags" : {
		"filename" : ["tags_test/{}.txt".format(i) for i in range(2000)],
		"data" : []
	}
}
    
#############################################################
##                                                         ##
##                                                         ##
##                                                         ##
##               DESCRIPTION PRE-PROCESSING                ##
##                                                         ##
##                                                         ##
##                                                         ##
#############################################################
def cleanDescriptionBOW(description: list, nouns_only: bool) -> dict:
	document_tokens, document_keywords = [], []
	for idx, text in enumerate(description):
		# Pre-processing
		text = text.lower() # to lower case
		text = text.strip() # strip white space
		text = re.sub(r'\d+', ' ', text) # remove digits
		text = re.sub(r'[^\w\s]', " ", text) # remove punctuation
		text = re.sub("[ |\t]{2,}", " ", text) # remove tabs

		# Tokenize
		if nouns_only:
			tokens = [
				item[0] \
				for item in nltk.pos_tag(nltk.word_tokenize(text)) \
				if item[1] == "NN"
			]
		else:
			tokens = nltk.word_tokenize(text)


		# Stem
		stemmer=nltk.stem.porter.PorterStemmer()
		tokens = [stemmer.stem(token) for token in tokens]

		# Remove stopwords
		stopwords = nltk.corpus.stopwords.words('english')
		tokens = [token for token in tokens if token not in stopwords]

		# Extract keywords
		r = Rake()
		r.extract_keywords_from_text(text)
		keywords = r.get_ranked_phrases()
		if nouns_only:
			keywords = [
				item[0]
			    for sublist in [
					nltk.pos_tag(nltk.word_tokenize(keyword))
					for keyword in keywords
				]
				for item in sublist if item[1]=="NN"
			]


		document_tokens += tokens
		document_keywords += keywords

	token_count, keyword_count = dict(Counter(document_tokens)), dict(Counter(document_keywords))

	document_bow = {**token_count, **keyword_count}

	return(document_bow)


def getDescriptionBOW(d_set: dict, description_idx: int, nouns_only: bool):
	with open(ROOT + d_set["descriptions"]["filename"][description_idx]) as f:
		raw_lines = f.readlines()

	return (cleanDescriptionBOW(description=raw_lines, nouns_only=nouns_only))
        
def getDescriptionBOW_TEST(d_set: dict, description_idx: int, nouns_only: bool):
	with open(ROOT_TEST + d_set["descriptions"]["filename"][description_idx]) as f:
		raw_lines = f.readlines()

	return (cleanDescriptionBOW(description=raw_lines, nouns_only=nouns_only))

##
## CREATE BOW ALL TRAIN
##
bow_all_train_dict_list = []
word_index_all_train = []
for i in range(10000):
	result = getDescriptionBOW(
		d_set=TRAIN_SET,
		description_idx=i,
		nouns_only=False
	)
	bow_all_train_dict_list.append(result)
	word_index_all_train += result.keys()

# Process word index
word_index_all_train = list(set(word_index_all_train))
bow_all_train = np.zeros((10000, len(word_index_all_train)))

for idx, bow_d in enumerate(bow_all_train_dict_list):
	for key in bow_d:
		bow_all_train[idx, word_index_all_train.index(key)] = bow_d[key]

##
## CREATE BOW NOUN TRAIN
##
bow_noun_train_dict_list = []
word_index_noun_train = []
for i in range(10000):
	result = getDescriptionBOW(
		d_set=TRAIN_SET,
		description_idx=i,
		nouns_only=True
	)
	bow_noun_train_dict_list.append(result)
	word_index_noun_train += result.keys()

# Process word index
word_index_noun_train = list(set(word_index_noun_train))

bow_noun_train = np.zeros((10000, len(word_index_noun_train)))
for idx, bow_d in enumerate(bow_noun_train_dict_list):
	for key in bow_d:
		bow_noun_train[idx, word_index_noun_train.index(key)] = bow_d[key]
       
##
## BOW of tags 
##
bow_all_tags = np.zeros((10000, 92))

for i in tag_dict.keys():
    a = -1
    for j in categories:
        a += 1
        if j in tag_dict[i]:
            bow_all_tags[i][a] += 1
    for j in sub_categories:
        a += 1
        if j in tag_dict[i]:
            bow_all_tags[i][a] += 1        

########################################################################################################################
## CREATE BOW ALL TEST
########################################################################################################################
bow_all_test_dict_list = []
word_index_all_test = []
for i in range(2000):
	result = getDescriptionBOW_TEST(
		d_set=TEST_SET,
		description_idx=i,
		nouns_only=False
	)
	bow_all_test_dict_list.append(result)
	word_index_all_test += result.keys()

# Process word index
word_index_all_test = word_index_all_train
bow_all_test = np.zeros((2000, len(word_index_all_test)))
for idx, bow_d in enumerate(bow_all_test_dict_list):
	for key in bow_d:
		try:
			bow_all_test[idx, word_index_all_test.index(key)] = bow_d[key]
		except Exception:
			pass


########################################################################################################################
## CREATE BOW NOUN TEST
########################################################################################################################
bow_noun_test_dict_list = []
word_index_noun_test = []
for i in range(2000):
	result = getDescriptionBOW_TEST(
		d_set=TEST_SET,
		description_idx=i,
		nouns_only=True
	)
	bow_noun_test_dict_list.append(result)
	word_index_noun_test += result.keys()

# Process word index
word_index_noun_test = word_index_noun_train
bow_noun_test = np.zeros((2000, len(word_index_noun_test)))
for idx, bow_d in enumerate(bow_noun_test_dict_list):
	for key in bow_d:
		try:
			bow_noun_test[idx, word_index_noun_test.index(key)] = bow_d[key]
		except Exception:
			pass
        
        
#############################################################
##                                                         ##
##                                                         ##
##                                                         ##
##               LOAD HIDDEN RESNET LAYERS                 ##
##                                                         ##
##                                                         ##
##                                                         ##
#############################################################
########################################################################################################################
## LOAD FC1000 TRAIN
########################################################################################################################
fc1000_train_raw = pd.read_csv(
	filepath_or_buffer=ROOT+"features_train/features_resnet1000_train.csv",
	header=None
)
fc1000_train_raw[0] = fc1000_train_raw[0].apply(lambda x: int(x.split('/')[1].replace('.jpg','')))
fc1000_train_raw_sorted = fc1000_train_raw.sort_values(by=[0])
fc1000_train_img = fc1000_train_raw_sorted[0].values
fc1000_train = fc1000_train_raw_sorted[list(range(1,1001))].values


########################################################################################################################
## LOAD FC1000 TEST
########################################################################################################################
fc1000_test_raw = pd.read_csv(
	filepath_or_buffer=ROOT_TEST+"features_test/features_resnet1000_test.csv",
	header=None
)
fc1000_test_raw[0] = fc1000_test_raw[0].apply(lambda x: int(x.split('/')[1].replace('.jpg','')))
fc1000_test_raw_sorted = fc1000_test_raw.sort_values(by=[0])
fc1000_test_img = fc1000_test_raw_sorted[0].values
fc1000_test = fc1000_test_raw_sorted[list(range(1,1001))].values


########################################################################################################################
## LOAD POOL5 TRAIN
########################################################################################################################
pool5_train_raw = pd.read_csv(
	filepath_or_buffer=ROOT+"features_train/features_resnet1000intermediate_train.csv",
	header=None
)
pool5_train_raw[0] = pool5_train_raw[0].apply(lambda x: int(x.split('/')[1].replace('.jpg','')))
pool5_train_raw_sorted = pool5_train_raw.sort_values(by=[0])
pool5_train_img = pool5_train_raw_sorted[0].values
pool5_train = pool5_train_raw_sorted[list(range(1,2049))].values


########################################################################################################################
## LOAD POOL5 TEST
########################################################################################################################
pool5_test_raw = pd.read_csv(
	filepath_or_buffer=ROOT_TEST+"features_test/features_resnet1000intermediate_test.csv",
	header=None
)
pool5_test_raw[0] = pool5_test_raw[0].apply(lambda x: int(x.split('/')[1].replace('.jpg','')))
pool5_test_raw_sorted = pool5_test_raw.sort_values(by=[0])
pool5_test_img = pool5_test_raw_sorted[0].values
pool5_test = pool5_test_raw_sorted[list(range(1,2049))].values

########################################################################################################################
## PCA on Bag of words - Tags
########################################################################################################################
from sklearn.decomposition import PCA

pca = PCA(n_components=1000)
pca.fit(bow_noun_train)
bow_noun_train2 = pca.transform(bow_noun_train)

pca = PCA(n_components=1000)
pca.fit(bow_noun_test)
bow_noun_test2 = pca.transform(bow_noun_test)

########################################################################################################################
## MLP Regression
########################################################################################################################

from sklearn.neural_network import MLPRegressor
#validation_fraction=0.1,
reg = MLPRegressor(hidden_layer_sizes=(75,75,75,75,75,),  activation='relu', solver='adam',  batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False,  beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

reg = reg.fit(bow_noun_train2, fc1000_train)

bow_predict_test = reg.predict(bow_noun_test2)

########################################################################################################################
## k-NN and ranking images
########################################################################################################################

from sklearn.neighbors import NearestNeighbors
import matplotlib.image as mpimg

def predict(trained_model, KNN_fit_data, x_to_predict, image_dir, img_index):
	nbrs = NearestNeighbors(n_neighbors=20).fit(KNN_fit_data)
	distances, indices = nbrs.kneighbors(
		trained_model.predict(x_to_predict)
	)

	

	return(indices)

result_images = []
	for img_filename in [img_index[i] for i in indices[0]]:
		result_images.append(mpimg.imread(image_dir+str(img_filename)+".jpg"))
	plt.figure(1)
	for idx, im in enumerate(result_images):
		plt.subplot(4, 5, idx+1)
		plt.imshow(result_images[idx])
	plt.show()
predict(
	trained_model=reg,
	KNN_fit_data=fc1000_test,
	x_to_predict=np.array([bow_predict_test[1]]),
	image_dir=ROOT_TEST+"/images_test/",
	img_index=list(range(2000))
)

########################################################################################################################
## REVERSE TRAINING: with pool5_train
########################################################################################################################

nbrs = NearestNeighbors(n_neighbors=20).fit(pool5_train)

distances, indices = nbrs.kneighbors(pool5_test)

#bow_all_intermediate = np.zeros((1, len(bow_all_train[1])))
bow_all_test_refreshed = np.zeros((2000, len(bow_all_train[1])))

bow_all_test_avg = np.zeros((2000, len(bow_all_train[1])))

#arr = np.array([1, 5, 2, 4, 2])
#arr.argsort()[-3:][::-1]

counter = -1
for idx in indices:
    counter += 1
    #damping = 1
    for j in idx:
        bow_all_test_avg[counter] += bow_all_train[j]
    bow_all_test_avg[counter] = bow_all_test_avg[counter] / 20
    
#test_sample = bow_all_test[19]

nbrs1 = NearestNeighbors(n_neighbors=20).fit(bow_all_test_avg)
#distances1, indices1 = nbrs1.kneighbors(test_sample)
distances3, indices3 = nbrs1.kneighbors(bow_all_test) 


image_dir=ROOT_TEST+"/images_test/"
img_index=list(range(2000))

result_images = []
for img_filename in [img_index[i] for i in indices1[0]]:
	result_images.append(mpimg.imread(image_dir+str(img_filename)+".jpg"))
	plt.figure(1)
    
for idx, im in enumerate(result_images):
	plt.subplot(4, 5, idx+1)
	plt.imshow(result_images[idx])
	plt.show()




    
submission = pd.DataFrame({
	"Descritpion_ID":["{}.txt".format(i) for i in list(range(2000))],
	"Top_20_Image_IDs":[" ".join(["{}.jpg".format(i) for i in  indx]) for indx in indices3]
})
submission.to_csv(
	"E:/Acads/Applied Machine Learning/Final Exam/pool5_reverse_features.csv",
	index=False
)
    

########################################################################################################################
## REVERSE TRAINING: with fc1000
########################################################################################################################

nbrs = NearestNeighbors(n_neighbors=20).fit(fc1000_train)
distances, indices = nbrs.kneighbors(fc1000_test)

#bow_all_intermediate = np.zeros((1, len(bow_all_train[1])))
bow_all_test_refreshed = np.zeros((2000, len(bow_all_train[1])))
bow_all_test_avg = np.zeros((2000, len(bow_all_train[1])))

counter = -1
for idx in indices:
    counter += 1
    for j in idx:
        bow_all_test_avg[counter] += bow_all_train[j]
    bow_all_test_avg[counter] = bow_all_test_avg[counter] / 20
    
#test_sample = bow_all_test[19]
#distances1, indices1 = nbrs1.kneighbors(test_sample)

nbrs1 = NearestNeighbors(n_neighbors=20).fit(bow_all_test_avg)
distances3, indices3 = nbrs1.kneighbors(bow_all_test) 

image_dir=ROOT_TEST+"/images_test/"
img_index=list(range(2000))

result_images = []
for img_filename in [img_index[i] for i in indices1[0]]:
	result_images.append(mpimg.imread(image_dir+str(img_filename)+".jpg"))
	plt.figure(1)
    
for idx, im in enumerate(result_images):
	plt.subplot(4, 5, idx+1)
	plt.imshow(result_images[idx])
	plt.show()

    
submission = pd.DataFrame({
	"Descritpion_ID":["{}.txt".format(i) for i in list(range(2000))],
	"Top_20_Image_IDs":[" ".join(["{}.jpg".format(i) for i in  indx]) for indx in indices3]
})
submission.to_csv(
	"E:/Acads/Applied Machine Learning/Final Exam/fc1000_all_reverse_features_v2.csv",
	index=False
)


    
'''  
for i in indices:
    bow_all_intermediate = np.zeros((1, len(bow_all_train[1])))
    #damping = 1
    for j in i:
        damping -= 0.05
        bow_all_intermediate += bow_all_train[i] 
'''
    

########################################################################################################################
## SVM Regression
########################################################################################################################

from sklearn import svm

X = [[0, 0], [2, 2]]
y = [[0.5,0.3], [2.5,1]]

clf = svm.SVR()

clf.fit(X, y) 

clf.predict([[1, 1]])


X = bow_noun_train
X1 = X.tolist()

y = fc1000_train
y1 = y.tolist()

clf = svm.SVR()

clf.fit(X1, y1) 

#SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
#    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

clf.predict([[1, 1]])