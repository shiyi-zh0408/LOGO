import pickle

with open('Annotations/fine-grained_annotation_aqa.pkl', 'rb') as f:
    data_anno = pickle.load(f)
with open('Annotations/FineDiving_coarse_annotation.pkl', 'rb') as f:
    FineDiving_coarse_annotation = pickle.load(f)
with open('Annotations/FineDiving_fine-grained_annotation.pkl', 'rb') as f:
    FineDiving_fine = pickle.load(f)

print(1)