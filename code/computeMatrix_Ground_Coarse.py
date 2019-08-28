import os

files = os.listdir("/u/scr/mhahn/deps/manual_output_ground_coarse_final/")
languages = []
for f in files:
   l = f[:f.index("_infer")]
   m = f.split("_")[-1][:-4]
   languages.append((l,m))
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
    command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "computeDependencyLengths_ForGrammars_RawUD.py", "--language="+lang, "--model="+mod, "--BASE_DIR=manual_output_ground_coarse_final"]
    print(" ".join(command))
    subprocess.Popen(command).wait()



