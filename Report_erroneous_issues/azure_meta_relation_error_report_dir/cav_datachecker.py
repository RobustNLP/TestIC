import pickle as pk

with open('coco-train-words.p', 'rb') as f:
    feature = pk.load(f)

print((feature))