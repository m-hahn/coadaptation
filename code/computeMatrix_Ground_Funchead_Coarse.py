with open("/juicier/scr120/scr/mhahn/CODE/grammar-optim/grammars/manual_output_funchead_ground_coarse_final/auto-summary-lstm.tsv", "r") as inFile:
    data = [x.split("\t") for x in inFile.read().strip().split("\n")]
header = data[0]
header = dict(zip(header, range(len(header))))
data = data[1:]
languages = list(set([(x[header["Language"]], x[header["FileName"]]) for x in data]))
import random
random.shuffle(languages)
print(languages)

langs = [x[0] for x in languages]
models = [x[1] for x in languages]

import random
pairs = [(x,y) for x in langs for y in models]
import subprocess
for lang, mod in pairs:
    print((lang, mod))
    subprocess.Popen(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "computeDependencyLengths_ForGrammars.py", "--language="+lang, "--model="+mod, "--BASE_DIR=manual_output_funchead_ground_coarse_final"]).wait()



