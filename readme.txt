-------final_feature_analysis.py------
In this file, I used ANOVA and Random forest methods on DNA methylation dataset to get a reduced set of features/CPG markers. 

Input: BetaData_27K_SimpleImpute_Mean_1.csv and BetaData_450K_SimpleImpute_Zero_1.

Output: 4 sets of reduced features: 1) 27_sim_AnovaRF_markers_woSMOTE, 2) 27_sim_AnovaRF_markers_wSMOTE, 3) 450K_sim_AnovaRF_markers_woSMOTE, 4) 450K_sim_AnovaRF_markers_wSMOTE.

Description: First, a csv file with methylation value is imported as a dataframe (df) using pandas library. The first column 'Donor_Sample' does not add any meaning to the model, so I dropped it. 
The df is split into features (X) and target arrays (Y) for analysis. SMOTE is applied to features for balancing the data using python's imblearn package. Then the value of target column 
is updated (target) as number of samples are different now after SMOTE application. 
ANOVA and Random forest models are applied both separately (only ANOVA, only RF) and together (ANOVA_RF) to get the reduced setS of features. This process was done by using Scikit-learn
package and it is done on both the balanced and imbalanced dataset.
Dataset is divided into 3 sets (train, validation and test) to verify the importance of reduced features on Random forest model. Both arrays of total features and reduced features are 
applied to Random forest model to see if reduced features perform well in classification. 
I also calculated number of common features in the list of only ANOVA and only RF. 
Finally, array of of reduced features (methylation value) and list of reduced features (CPG markers) are converted to a csv file for further analysis.

------annotation_final.py-----
In this file, I compared the closest genes associated with reduced markers with the cancer related genes collected from several resources (COSMIC, TSGene database, etc).

Input: CSV file containing sets of genes in column 'Gene_Symbol'.

Output: Common genes between the resources and our reduced sets of genes.

Description: Genes from COSMIC, TSGene, etc were collected and python's intersection method was used to find common genes among different sets.


------EA.R-------
I performed GSEA analysis on the closest genes associated with reduced markers.
Input: CSV file containing sets of genes in column 'Gene_Symbol'.

Output: GSEA figures for all 4 sets of genes.

Description: R's TCGAbiolinks package is used for GSEA analysis. TCGAanalyze_EAcomplete function is used on sets of genes to identify classes of genes or proteins that are over-represented using annotations for that gene set. Then 
TCGAvisualize_EAbarplot was used to show canonical pathways significantly overrepresented (enriched) by the DEGs (differentially expressed genes) with the number of genes for the main categories of three ontologies 
(GO:biological process, GO:cellular component, and GO:molecular function, respectively).