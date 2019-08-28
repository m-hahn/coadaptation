with open("/juicier/scr120/scr/mhahn/CODE/grammar-optim/grammars/manual_output_funchead_ground_coarse_final/auto-summary-lstm.tsv", "r") as inFile:
    data = [x.split("\t") for x in inFile.read().strip().split("\n")]
header = data[0]
header = dict(zip(header, range(len(header))))
data = data[1:]
languages = list(set([(x[header["Language"]], x[header["FileName"]]) for x in data]))
import random
languages = sorted(languages)
print(languages)

langs = [x[0] for x in languages]

import random
pairs = [(x,y,z) for x in langs for y,z in languages]
import subprocess
with open("output/dlm_ground_funchead_coarse.tsv", "w") as out:
   print("\t".join(["Language", "GrammarLanguage", "Grammar", "Length"]), file=out)
   for lang, lang2, mod in pairs:
      outpath = "../raw_results/funchead_coarse/raw/"+lang+"_"+mod
      #print(outpath)
      try:
   
         length = float(open(outpath, "r").read().strip())
      except IOError:
         continue
      print("\t".join([str(x) for x in [lang, lang2, mod, length]]), file=out)


