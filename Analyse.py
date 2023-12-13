import pandas as pd

# read
df = pd.read_parquet('../data/run_ww_2019_d.parquet')
count=0
print("1-------------------------------------")
print(f"Nom des colonnes: {df.columns}")
# print("2-------------------------------------")
# print(f"Nombre de lignes: {len(df)}")
# print("3-------------------------------------")
# print(df.distance.describe())
# print("4-------------------------------------")
# print(df.athlete)
print("5-------------------------------------")
print(len(df.loc[df.athlete == 37594]))
print(df.loc[df.athlete == 37594])
print("6-------------------------------------")
print(df.loc[df.distance > 42].duration)
print("7-------------------------------------")
print(df.loc[df.distance == 42].duration.describe())


for train in df:
    # print("train : " + train.distance)
    print("train : " + train)
    if train.loc[train.athlete == 37594]:
        if train.loc[df.distance == 0]:
                count += 1


print("compteur distance == 0 : " + str(count))