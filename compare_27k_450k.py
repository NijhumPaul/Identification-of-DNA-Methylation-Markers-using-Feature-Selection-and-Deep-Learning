import pandas as pd
df_27 = pd.read_csv("BetaData_27K_SimpleImpute_Mean_1.csv")
df_450 = pd.read_csv("BetaData_450K_SimpleImpute_Zero_1.csv")
markers_27=set(list(df_27.columns)[2:-1])
print("27k total markers:",len( markers_27))
markers_450=list(df_450.columns)[2:-1]
print("450k total markers:", len(markers_450))
common_merkers=markers_27.intersection(markers_450)
print(len(common_merkers))
