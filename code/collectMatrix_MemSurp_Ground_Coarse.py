import os
files = os.listdir("/u/scr/mhahn/deps/manual_output_ground_coarse_final/")
languages = []
for f in files:
   l = f[:f.index("_infer")]
   m = f.split("_")[-1][:-4]
   languages.append((l,m))
import random
languages = sorted(languages)
print(languages)

langs = [x[0] for x in languages]

import random
pairs = [(x,y,z) for x in langs for y,z in languages]
import subprocess
with open("output/memsurp_ground_coarse.tsv", "w") as out:
   print("\t".join(["Language", "GrammarLanguage", "Grammar", "Length"]), file=out)
   for lang, lang2, mod in pairs:
      outpath = "../raw_results/ud_coarse/raw_memsurp/"+lang+"_"+mod
      #print(outpath)
      try:
   
         length = float(open(outpath, "r").read().strip())
      except IOError:
         continue
      print("\t".join([str(x) for x in [lang, lang2, mod, length]]), file=out)


