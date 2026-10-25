import pandas as pd

d = "data/raw/ruhsold/"
for name in ["Fine_Grained_train", "Fine_Grained_test", "Coarse_Grained_train", "Coarse_Grained_test"]:
    df = pd.read_csv(d + name + ".csv")
    print(f"\n{name}: rows={len(df)} null_labels={df.label.isna().sum()} dtype={df.label.dtype}")
    print(df.label.value_counts(dropna=False).sort_index().to_string())