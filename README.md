# Dcirc_m6A
a tool for circRNA_m6A predicting.We have developed a method based on deep learning to identify m6A in circrna.Finally, the accuracy rate of our training model on the test data set is more than 90%. We have developed it into a prediction tool. You can directly use our trained model to carry out prediction tasks directly, which is very convenient; or you can use your own data to train new models.  

# Depetency
Python3.7 pysam>=0.15 biopython>=1.78 pandas>=0.24 numpy>=1.18 sklearn>=0.22 pytorch>=1.5  

# Usage
## 1. Pretreatment
### 1.1 mapping result handle
If you only have the raw data of sequencing, you need to use bowtie2 and hisat2 to compare the software with the reference genome to obtain the SAM file.
