'if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")'

#install.packages('devtools')
#devtools::install_github(repo = "BioinformaticsFMRP/TCGAbiolinks")
#install.packages("TCGAbiolinks")
BiocManager::install("clusterProfiler")
BiocManager::install("org.Hs.eg.db")
#BiocManager::install("EDASeq")


library(TCGAbiolinks)
library(EDASeq)
library(clusterProfiler)
library(org.Hs.eg.db)

df = read.csv("C:/Users/nijhu/OneDrive - North Dakota University System/COBRE/project_COBRE/data/genes_long_format/sure/27_sim_AnovaRF_markerData_wSMOTE.csv", header=TRUE)
df_all = read.csv("C:/Users/nijhu/OneDrive - North Dakota University System/COBRE/project_COBRE/data/genes_long_format/sure/27k_All.csv", header=TRUE)

print(length(df$Gene_Symbol))
Genelist_reduced <- unique(df$Gene_Symbol)
gene_all <- unique(df_all$Gene_Symbol)
print(length(Genelist_reduced))
print(length(gene_all))

#gene_symbols<-gene_DEG_102651_Merged$Gene
entrez_IDS <- mapIds(org.Hs.eg.db, Genelist_reduced, 'ENTREZID', 'SYMBOL')
entrez_IDS_all <- mapIds(org.Hs.eg.db, gene_all, 'ENTREZID', 'SYMBOL')
entrez_gene<-data.frame(keyName=names(entrez_IDS), value=entrez_IDS, row.names=NULL) 

go_DEG_BP <- enrichGO(gene = entrez_IDS,
                            universe = entrez_IDS_all,
                            OrgDb = org.Hs.eg.db,
                            ont = "CC",
                            pAdjustMethod = "BH",
                            pvalueCutoff = 0.5,
                            qvalueCutoff = 0.5,
                            readable = TRUE
                            )

cluster_summary_DEG <- data.frame(go_DEG_BP)
cluster_summary_DEG
write.csv(cluster_summary_DEG,"GO_DEG_MF_27k_SMOTE.csv")
