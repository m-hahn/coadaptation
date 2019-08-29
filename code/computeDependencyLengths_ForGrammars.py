
import random
import sys

from collections import deque


import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--temperature", type=str, default="Infinity")
parser.add_argument("--BASE_DIR", type=str)
parser.add_argument("--printDetailedData", type=bool)

args=parser.parse_args()

assert args.temperature == "Infinity"

myID = random.randint(0,10000000)

posUni = set() 
posFine = set() 



from math import log, exp
from random import random, shuffle
import os


my_fileName = __file__.split("/")[-1]

from corpusIterator_FuncHead import CorpusIteratorFuncHead

originalDistanceWeights = {}


def makeCoarse(x):
   if ":" in x:
      return x[:x.index(":")]
   return x

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIteratorFuncHead(args.language,partition).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          line["coarse_dep"] = makeCoarse(line["dep"])
          depsVocab.add(line["coarse_dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          if line["coarse_dep"] == "root":
             continue
          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["coarse_dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = dep
          keyWithDir = (dep, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   #print orderTable
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key, "HD"), 0) + 1.0
      dh = orderTable.get((key, "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab

#import torch.distributions
#import torch.nn as nn
#import torch
#from torch.autograd import Variable


# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   allGradients = gradients_from_the_left_sum # + sum(line.get("children_decisions_logprobs",[]))
   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients

import numpy.random

#softmax_layer = torch.nn.Softmax()
#logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       logits = [(x, distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       childrenLinearized = list(map(lambda x:x[0], logits))
       return childrenLinearized           

def orderSentence(sentence, dhLogits, printThings):
   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   for line in sentence:
      line["coarse_dep"] = makeCoarse(line["dep"])
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      dhSampled = (dhLogit > 0) #(random() < probability.data.numpy())
      
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print("\t".join(list(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])



   for line in sentence:
      if "children_DH" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
         line["children_DH"] = childrenLinearized
      if "children_HD" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
         line["children_HD"] = childrenLinearized

   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, 0)
   if printThings:
     print(" ".join(list(map(lambda x:x["word"], sentence))))
     print(" ".join(list(map(lambda x:x["word"], linearized))))


   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
      moved[x["index"]-1] = i
   for i,x in enumerate(linearized):
      if x["head"] == 0: # root
         x["reordered_head"] = 0
      else:
         x["reordered_head"] = 1+moved[x["head"]-1]
   return linearized, logits


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()

posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

posFine = list(posFine)
itos_pos_ptb = posFine
stoi_pos_ptb = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

print(itos_deps)

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)



if args.model != "RANDOM":
   inpModels_path = "/u/scr/mhahn/deps/"+args.BASE_DIR+"/"
   models = os.listdir(inpModels_path)
   models = list(filter(lambda x:"_"+args.model+".tsv" in x, models))
   if len(models) == 0:
     assert False, "No model exists"
   if len(models) > 1:
     assert False, [models, "Multiple models exist"]
   
   with open(inpModels_path+models[0], "r") as inFile:
      data = list(map(lambda x:x.split("\t"), inFile.read().strip().split("\n")))
      header = data[0]
      data = data[1:]
    
   if "CoarseDependency" not in header:
     header[header.index("Dependency")] = "CoarseDependency"
   if "DH_Weight" not in header:
     header[header.index("DH_Mean_NoPunct")] = "DH_Weight"
   if "DistanceWeight" not in header:
     header[header.index("Distance_Mean_NoPunct")] = "DistanceWeight"

   for line in data:
      dependency = line[header.index("CoarseDependency")]
      key = dependency
      if key not in stoi_deps:
         continue
      dhWeights[stoi_deps[key]] = float(line[header.index("DH_Weight")])
      distanceWeights[stoi_deps[key]] = float(line[header.index("DistanceWeight")])

      if "Counter" in header:
        originalCounter = int(line[header.index("Counter")])
      else:
        originalCounter = 200000

words = list(vocab.items())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = list(map(lambda x:x[0], words))
stoi = dict(zip(itos, range(len(itos))))

if len(itos) > 6:
   assert stoi[itos[5]] == 5

vocab_size = 50000

batchSize = 1

lr_lm = 0.1


crossEntropy = 10.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < vocab_size else 1



#import torch.cuda
#import torch.nn.functional




counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


def doForwardPass(current):
       global counter
       global crossEntropy
       global printHere
       global devLosses

       assert len(current) == 1, len(current)
       batchOrdered, logits = orderSentence(current[0], dhLogits, printHere)
       batchOrdered = [batchOrdered]
       logits = [logits]
#       batchOrderedLogits = list(zip(*list(map(lambda z:orderSentence(z[1], dhLogits, z[0]==0 and printHere), zip(range(len(current)),current)))))
      
 #      batchOrdered = batchOrderedLogits[0]
 #      logits = batchOrderedLogits[1]
   
       lengths = list(map(len, current))
       # current is already sorted by length
       maxLength = lengths[int(0.8*batchSize)]
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append([2 if i == 0 else (encodeWord(x[i-1]["word"]) if i <= len(x) else 0) for x in batchOrdered])
          input_pos_u.append(list(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered)))
          input_pos_p.append(list(map(lambda x: 2 if i == 0 else (stoi_pos_ptb[x[i-1]["posFine"]]+3 if i <= len(x) else 0), batchOrdered)))
       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()))
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0

       totalQuality = 0.0

       if True:
           totalDepLength = 0
           byType = []
           for i in range(1,len(input_words)): #range(1,maxLength+1): # don't include i==0
              for j in range(batchSize):
                 if input_words[i][j] != 0:
                    if batchOrdered[j][i-1]["head"] == 0:
                       realHead = 0
                    else:
                       realHead = batchOrdered[j][i-1]["reordered_head"] # this starts at 1, so just right for the purposes here
                    if batchOrdered[j][i-1]["coarse_dep"] == "root":
                       continue
                    depLength = abs(i - realHead) # - 1 # to be consistent with Richard's stuff
                    assert depLength >= 1
                    totalDepLength += depLength
                    byType.append((batchOrdered[j][i-1]["coarse_dep"], depLength))
                    if input_words[i][j] > 2 and j == 0 and printHere:
                       print([itos[input_words[i][j]-3], itos_pos_ptb[input_pos_p[i][j]-3] ])
                    wordNum += 1

       if wordNum > 0:
          crossEntropy = 0.99 * crossEntropy + 0.01 * (totalDepLength/wordNum)
       else:
          assert totalDepLength == 0
       numberOfWords = wordNum
       return (totalDepLength, numberOfWords, byType)



assert batchSize == 1

depLengths = []
if True:
  corpus = CorpusIteratorFuncHead(args.language,"train")
  corpusIterator = corpus.iterator()
  if corpus.length() == 0:
     quit()
  while True:
    try:
       batch = [next(corpusIterator)]
    except StopIteration:
       break
    partitions = range(1)
    
    for partition in partitions:
       counter += 1
       printHere = (counter % 200 == 0)
       current = batch[partition*batchSize:(partition+1)*batchSize]
       if len(current) == 0:
          continue
       depLength = doForwardPass(current)
       depLengths.append(depLength)
       if counter % 100 == 0:
          print("Average dep length "+str(sum(list(map(lambda x:x[0], depLengths)))/float(len(depLengths)))+" "+str(counter)+" "+str(float(counter)/corpus.length())+"%")

outpath = "../raw_results/funchead_coarse/raw/"+args.language+"_"+args.model
print(outpath)
with open(outpath, "w") as outFile:
   print(sum(list(map(lambda x:x[0], depLengths)))/float(len(depLengths)), file=outFile)


