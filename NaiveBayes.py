# -*- coding: utf-8 -*-
import os
import math
import random
import copy
import re
import time
from collections import defaultdict
global class_lst, filename, folder_path, group

start = time.time()
folder_path = 'D:/UTA/Fall2019/ML/Project2/20_newsgroups/'

def preprocess_text(text):
    re.sub(' +', ' ', text)
    replace_chars = ['"',"'",'!','/','\\','=',',',':','\n','<','>','?','.','"',')','(','|','-','#','*','+','~','{','}','[',']','@','%','@','&','^','`']    
    text = text.lower()
    for i in replace_chars:
        text = text.replace(i, ' ')    
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
                  "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
                  "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
                  "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", 
                  "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", 
                  "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", 
                  "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", 
                  "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", 
                  "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", 
                  "now"]    
    for j in stop_words:
        text = text.replace(j, '')
    return text


def probability(words, dictnry):
    dict_sum = sum(dictnry.values())
    prob = 0.0
    for word in words:
        value = dictnry.get(word, 0.0) + 0.0001
        prob = prob + math.log(float(value)/float(dict_sum))
    return prob

def fetch_file():
    global group
    while(len(class_lst)):
        ran_class = random.randint(0, len(class_lst) - 1)
        n_class = class_lst[ran_class]
        if len(filename[n_class]) == 0:
            class_lst.remove(n_class)
        else:
            rand_file = random.randint(0, len(filename[n_class]) - 1)
            class_file = filename[n_class][rand_file]
            filename[n_class].remove(class_file)
            group = n_class
            data = open(folder_path + n_class + '/' + class_file, 'r')
            return data.read()
    group = 'NULL'
    return 'NULL'

def kmeans(clas, k, maxiter=100):

    featr = dict((c, [c]) for c in clas[:k])
    featr[clas[k-1]] += clas[k:]
    for i in range(maxiter):
        new_featr = assign(featr)
        #print(new_centers)
        if featr == new_featr:
            break
        else:
            featr = new_featr
    return featr

def assign(featrs):
    new_featrs = defaultdict(list)
    for cx in featrs:
        for x in featrs[cx]:
            best = min(featrs)
            new_featrs[best] += [x]
    return new_featrs

training_data = 500
class_list = os.listdir(folder_path)
counter = 0
dict_total ={}
filename = {}
dictionary = {}
group = 'NULL'
print("Training the dataset........")

for clss in class_list:
    dictnry = {}
    folder = folder_path + clss
    files = os.listdir(folder)
    iteration = 0
    for file in files:
        iteration = iteration + 1
        if iteration > training_data:
            break
        filepath = folder + '/'+file
        myfile = open(filepath, 'r')
        data = preprocess_text(myfile.read())
        words = data.split(' ')
        for word in words:
            if word == ' ' or word == '':
                continue
            add_dict = dictnry.get(word, 0)
            add_dict_total = dict_total.get(word, 0)
            if add_dict_total == 0:
                dict_total[word] = 0
            else:
                dict_total[word] = add_dict_total + 1
            if add_dict == 0:
                dictnry[word] = 1
            else:
                dictnry[word] = add_dict + 1
            
        files.remove(file)      #iterating files
    filename[clss] = files
    dictionary[clss] = dictnry
print(len(dict_total), "words found in all the class files")
print("Testing the dataset........")
data = 1
class_lst = copy.deepcopy(class_list)
iteratn = 0
success = 0
while(data):
    data = fetch_file()
    iteratn = iteratn + 1
    if data == 'NULL':
        break
    data = preprocess_text(data)
    words = data.split(' ')
    if ' ' in words: words.remove(' ')
    if '' in words: words.remove('')
    probabilities = []
    for c_lst in class_list:
        probabilities.append(probability(words, dictionary[c_lst]))
    if group == class_list[probabilities.index(max(probabilities))]:
        success = success + 1
end = time.time()
print ('Success rate = %.1f'% (float(success)/float(iteratn - 1)*100), ", Time duration: " + str(end - start))
