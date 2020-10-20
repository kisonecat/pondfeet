import numpy as np
from bdflib import reader
from bdflib import writer
import math
import re
import os
import pickle

def data_to_array(data):
    rows = []
    for i in data:
        row = bin(int(i,16))[2:].zfill(len(i)*4)
        row = row.replace('1','#').replace('0','.')
        rows.append(row)
    return rows

def data_to_numpy(data):
    n = data_to_array(data)
    return np.array([[0 if x == '.' else 1 for x in row] for row in n],np.int8)

maxWidth = 6
maxHeight = 13

def process(handle,xshift=0,yshift=0):
    font = reader.read_bdf(handle)
    glyphs = {}
    if ord('A') in font:
        letter_a = font[ord('A')]
        bbW = letter_a.bbW
        bbY = letter_a.bbY
        bbX = letter_a.bbX
        bbH = letter_a.bbH
        advance = letter_a.advance
        print("size",bbW,bbH)
        
        if (bbW <= maxWidth) and (bbH <= maxHeight):
            for c in range(32,128):
                if c in font:
                    letter = font[c]
                    xs = data_to_numpy(letter.get_data())
                    if len(xs.shape) > 1:
                        xs = np.pad( xs, ((maxHeight+1,yshift),(xshift,maxWidth+1)),'constant', constant_values=(0, 0))
                        xs = xs[-maxHeight:,:maxWidth]
                        glyphs[c] = xs
    return glyphs

ROOT = 'fonts'
directory = os.fsencode(ROOT)

data = {}

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".bdf"):
         with open(os.path.join(ROOT,filename),"rb") as handle:
             print("Reading " + filename + "...")
             output = process(handle)
             if output != {}:
                 data[filename] = output
         with open(os.path.join(ROOT,filename),"rb") as handle:                 
             output = process(handle,xshift=1)
             if output != {}:
                 data[filename + "-xshift"] = output
         with open(os.path.join(ROOT,filename),"rb") as handle:
             output = process(handle,yshift=1)
             if output != {}:
                 data[filename + "-yshift"] = output
         continue
     else:
         continue

print("Saving...")
pickle.dump(data,open("fonts.pkl","wb"))
