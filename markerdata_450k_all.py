import pandas as pd

df = pd.read_csv("MarkerData_450K_All.csv")
df = df.replace({'.': 'OpenSea'})
gene_symbol = df['Gene_Symbol']
Composite_Element_REF = df['Composite Element REF']
total_symbol = []

for i in gene_symbol:
    i = i.replace(';', ',')
    i = i.split(",")
    i = list(set(i))
    values = len(i)
    #print(i)
    total_symbol.append(values)
max_length = max(total_symbol)
new_geneSymbol = []
for i in gene_symbol:
    i = i.replace(';', ',')
    i = i.split(",")
    i = list(set(i))
    print(i)
    i.extend([0] * (max_length - len(i)))
    new_geneSymbol.append(i)
#print(new_geneSymbol[0:10])
new_df = pd.DataFrame(new_geneSymbol)
#print(new_df.head(10))
column_names = []
for i in range (max_length):
    columns = "Gene_Symbol_" + str(i)
    column_names.append(columns)
new_df.columns = column_names
df1 = df.drop(['Gene_Symbol'], axis=1)
output_df = pd.concat([df1, new_df], axis=1)
print(output_df.shape)
output_df = output_df.rename(columns={"Composite Element REF": "Composite_Element_REF"})

output_df.to_csv("MarkerData_27K_All_Modified.csv")
'''gene_symbols = []
for i in gene_symbol:
    i = i.replace(';', ',')
    i = i.split(",")
    i = list(set(i))
    #print(i)
    gene_symbols.append(i)
#print(gene_symbols)
gene_symbol_df = pd.DataFrame(gene_symbols)
print(gene_symbol_df)'''
