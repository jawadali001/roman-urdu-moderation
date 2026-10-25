import pandas as pd

d = "data/raw/ruhsold/"
tr = pd.read_csv(d + "Fine_Grained_train.csv")
va = pd.read_csv(d + "Fine_Grained_validation.csv")
te = pd.read_csv(d + "Fine_Grained_test.csv")
cva = pd.read_csv(d + "Coarse_Grained_validation.csv")

print("sizes train/val/test:", len(tr), len(va), len(te))
print("fine val identical to train:", tr.equals(va))
print("fine val tweets also in train: %.1f%%" % (100 * va.tweet.isin(tr.tweet).mean()))
print("fine test tweets also in train: %.1f%%" % (100 * te.tweet.isin(tr.tweet).mean()))
print("coarse val tweets also in fine train: %.1f%%" % (100 * cva.tweet.isin(tr.tweet).mean()))
print("\nfine train label counts (ids):")
print(tr.label.value_counts().sort_index())