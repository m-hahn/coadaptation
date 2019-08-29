#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys



import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--temperature", type=str, default="Infinity")
parser.add_argument("--BASE_DIR", type=str)

args=parser.parse_args()

assert args.temperature == "Infinity"





posUni = set() 
posFine = set() 



from math import log, exp, sqrt
from random import random, shuffle, randint
import os


from corpusIterator import CorpusIterator

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
     for sentence in CorpusIterator(args.language,partition).iterator():
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
import torch.nn as nn
import torch
from torch.autograd import Variable


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

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



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



crossEntropy = 10.0

def encodeWord(w, doTraining):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

def regularisePOS(w, doTraining):
   return w



import torch.cuda
import torch.nn.functional


baselineAverageLoss = 0

counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 
devLossesWords = []
devLossesPOS = []

loss_op = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index = 0)

bigramCounts = {"PLACEHOLDER" : {}}
unigramCountsL = {"PLACEHOLDER" : len(itos)}
unigramCountsR = {"PLACEHOLDER" : len(itos)}

for word in itos:
    unigramCountsL[word] = 1
    unigramCountsR[word] = 1
    bigramCounts[word] = {"PLACEHOLDER" : 1}
    bigramCounts["PLACEHOLDER"][word] = 1

bigramCountsPOSFine = {"PLACEHOLDER" : {}}
unigramCountsPOSFineL = {"PLACEHOLDER" : len(posFine)}
unigramCountsPOSFineR = {"PLACEHOLDER" : len(posFine)}

for word in posFine:
    unigramCountsPOSFineL[word] = 1
    unigramCountsPOSFineR[word] = 1
    bigramCountsPOSFine[word] = {"PLACEHOLDER" : 1}
    bigramCountsPOSFine["PLACEHOLDER"][word] = 1






def doForwardPassTrain(current, train=True):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       global baselineAverageLoss
       assert len(current) == 1, len(current)
       batchOrdered, logits = orderSentence(current[0], dhLogits, printHere)
       batchOrdered = [batchOrdered]
       logits = [logits]
   
       lengths = list(map(len, current))
       # current is already sorted by length
       maxLength = max(lengths)
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"], train) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 if i == 0 else (regularisePOS(stoi_pos_ptb[x[i-1]["posFine"]]+3, train) if i <= len(x) else 0), batchOrdered))
       posFines = ["SOS"] + [x["posFine"] for x in batchOrdered[0]] + ["EOS"]
       for i in range(len(posFines)-1):
         left = posFines[i]
         right = posFines[i+1]
         if left not in bigramCountsPOSFine:
            bigramCountsPOSFine[left] = {}
         bigramCountsPOSFine[left][right] = bigramCountsPOSFine[left].get(right,0)+1
         unigramCountsPOSFineL[left] = unigramCountsPOSFineL.get(left, 0)+1
         unigramCountsPOSFineR[right] = unigramCountsPOSFineR.get(right, 0)+1
 
       words = ["SOS"] + [x["word"] for x in batchOrdered[0]] + ["EOS"]
       for i in range(len(words)-1):
         left = words[i]
         right = words[i+1]
         if left not in bigramCounts:
            bigramCounts[left] = {}
         bigramCounts[left][right] = bigramCounts[left].get(right,0)+1
         unigramCountsL[left] = unigramCountsL.get(left, 0)+1
         unigramCountsR[right] = unigramCountsR.get(right, 0)+1
 
from math import log

def doForwardPassEvaluate(current, train=True):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       global baselineAverageLoss
       assert len(current) == 1, len(current)
       batchOrdered, logits = orderSentence(current[0], dhLogits, printHere)
       batchOrdered = [batchOrdered]
       logits = [logits]
   
       lengths = list(map(len, current))
       # current is already sorted by length
       maxLength = max(lengths)
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"], train) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 if i == 0 else (regularisePOS(stoi_pos_ptb[x[i-1]["posFine"]]+3, train) if i <= len(x) else 0), batchOrdered))



       posFines = ["SOS"] + [x["posFine"] for x in batchOrdered[0]] + ["EOS"]
       surprisalPOS = 0
       delta = 0.5
       for i in range(len(posFines)-1):
         left = posFines[i]
         right = posFines[i+1]
         bigramCountsPOSFineLeft = bigramCountsPOSFine.get(left, {})
         bigramCount = bigramCountsPOSFineLeft.get(right, 0)

         #print(left, bigramCountsPOSFineLeft, bigramCount)

         unigramCountsPOSFineLLeft = unigramCountsPOSFineL.get(left, 0)
         unigramCountsPOSFineRRight = unigramCountsPOSFineR.get(right, 0)
         prob = (max(bigramCount-delta, 0.0) + float(unigramCountsPOSFineRRight)/totalUnigramCount * delta * len(bigramCountsPOSFineLeft))/unigramCountsPOSFineLLeft
#         totalProb = 0
#         for posFine in itos + ["PLACEHOLDER"]:
#             bigramCount1 = bigramCountsPOSFineLeft.get(posFine, 0)
#             unigramCountsPOSFineRRight1 = unigramCountsPOSFineR.get(posFine, 0)
#             prob2 = (max(bigramCount1-delta, 0.0) + float(unigramCountsPOSFineRRight1)/totalUnigramCount * delta * len(bigramCountsPOSFineLeft))/unigramCountsPOSFineLLeft
#             totalProb += prob2
#         print(totalProb)
#         assert totalProb <= 1.01, (totalProb, left)
         assert prob <= 1.0
         surprisalPOS -= log(prob)
         if printHere and i > 0 and i < len(batchOrdered[0]):
             print("\t".join(map(str,[batchOrdered[0][i]["posFine"], batchOrdered[0][i]["posFine"], log(prob), log(float(unigramCountsPOSFineRRight)/totalUnigramCount)])))



       words = ["SOS"] + [x["word"] for x in batchOrdered[0]] + ["EOS"]
       surprisalWord = 0
       delta = 0.5
       for i in range(len(words)-1):
         left = words[i]
         right = words[i+1]
         bigramCountsLeft = bigramCounts.get(left, {})
         bigramCount = bigramCountsLeft.get(right, 0)

         #print(left, bigramCountsLeft, bigramCount)

         unigramCountsLLeft = unigramCountsL.get(left, 0)
         unigramCountsRRight = unigramCountsR.get(right, 0)
         prob = (max(bigramCount-delta, 0.0) + float(unigramCountsRRight)/totalUnigramCount * delta * len(bigramCountsLeft))/unigramCountsLLeft
#         totalProb = 0
#         for word in itos + ["PLACEHOLDER"]:
#             bigramCount1 = bigramCountsLeft.get(word, 0)
#             unigramCountsRRight1 = unigramCountsR.get(word, 0)
#             prob2 = (max(bigramCount1-delta, 0.0) + float(unigramCountsRRight1)/totalUnigramCount * delta * len(bigramCountsLeft))/unigramCountsLLeft
#             totalProb += prob2
#         print(totalProb)
#         assert totalProb <= 1.01, (totalProb, left)
         assert prob <= 1.0
         surprisalWord -= log(prob)
         if printHere and i > 0 and i < len(batchOrdered[0]):
             print("\t".join(map(str,[batchOrdered[0][i]["word"], batchOrdered[0][i]["posFine"], log(prob), log(float(unigramCountsRRight)/totalUnigramCount)])))
       _ = 0
       return _, _, _, surprisalWord+surprisalPOS, len(words)+1,surprisalWord, surprisalPOS


def computeDevLoss():
   global printHere
   global counter
   devLoss = 0.0
   devLossWords = 0.0
   devLossPOS = 0.0
   devWords = 0
#   corpusDev = getNextSentence("dev")
   corpusDev = CorpusIterator(args.language,"dev").iterator()

   while True:
     try:
        batch = [next(corpusDev)]
     except StopIteration:
        break
     partitions = range(1)
     for partition in partitions:
        counter += 1
        printHere = (counter % 1000 == 0)
        current = batch[partition*batchSize:(partition+1)*batchSize]
 
        _, _, _, newLoss, newWords, lossWords, lossPOS = doForwardPassEvaluate(current, train=False)

        devLoss += newLoss
        devWords += newWords
        devLossWords += lossWords
        devLossPOS += lossPOS
   return devLoss/devWords, devLossWords/devWords, devLossPOS/devWords

if True:
  corpus = CorpusIterator(args.language).iterator()


  while True:
    try:
       batch = [next(corpus)]
    except StopIteration:
       break
    partitions = range(1)
    for partition in partitions:
       counter += 1
       printHere = (counter % 1000 == 0)
       current = batch[partition*batchSize:(partition+1)*batchSize]

       doForwardPassTrain(current)
  #print(bigramCounts)

  totalUnigramCount = sum([y for x,y in unigramCountsR.items()])

  if True: #counter % 10000 == 0:
          newDevLoss, newDevLossWords, newDevLossPOS = computeDevLoss()
          devLosses.append(newDevLoss)
          devLossesWords.append(newDevLossWords)
          devLossesPOS.append(newDevLossPOS)
          print(devLossesWords)

outpath = "../raw_results/ud_coarse/raw_memsurp/"+args.language+"_"+args.model
print(outpath)
with open(outpath, "w") as outFile:
   print(newDevLossWords, file=outFile)


