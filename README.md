DNA methylation is a process that can affect gene accessibility and therefore gene expression. Methylation can affect genes that are associated with suppressing or contributing to tumor growth and progression. I used DNA methylation dataset to develop cancer prediction tool and identify biomarkers. The TCGA-BRCA dataset is used for this analysis. I have used two feature engineering methods (ANOVA F-Test and Random Forest) to reduce the dimensionality so that our deep learning model works better on the dataset. Also, normalizing the data and handling the data imbalance has proved to increase the model's performance much better. SMOTE technique was used to handle the oversampling in the dataset. Prediction using 450K methylation markers can be accomplished in less than 13 s with an accuracy of 98.75%. Of the list of 685 genes in the feature selected 27K dataset, 578 were mapped to Ensemble Gene IDs. This reduced set was significantly (FDR < 0.05) enriched in five biological processes and one molecular function. Of the list of 1572 genes in the feature selected 450K data set, 1290 were mapped to Ensemble Gene IDs. This reduced set was significantly (FDR < 0.05) enriched in 95 biological processes and 17 molecular functions. Seven oncogene/tumor suppressor genes were common between the 27K and 450K feature selected gene sets. These genes were RTN4IP1, MYO18B, ANP32A, BRF1, SETBP1, NTRK1, and IGF2R.

<img width="1878" height="902" alt="image" src="https://github.com/user-attachments/assets/d6abeb38-51fa-488e-a230-84f65f6acd8d" />

<img width="2616" height="5304" alt="image" src="https://github.com/user-attachments/assets/de148a76-b001-44fc-8ab3-db0f5a614344" />

<img width="2624" height="1780" alt="image" src="https://github.com/user-attachments/assets/14facc06-6de7-43cd-8189-7babae3fa541" />



