# Dcirc_m6A
a tool for circRNA_m6A predicting.We have developed a method based on deep learning to identify m6A in circrna.Finally, the accuracy rate of our training model on the test data set is more than 90%. We have developed it into a prediction tool. You can directly use our trained model to carry out prediction tasks directly, which is very convenient; or you can use your own data to train new models.  

# Depetency
Python3.7 pysam>=0.15 biopython>=1.78 pandas>=0.24 numpy>=1.18 sklearn>=0.22 pytorch>=1.5  

# Usage
## 1. Pretreatment
### 1.1 mapping result handle
If you only have the raw data of sequencing, you need to use bowtie2 and hisat2 to compare the software with the reference genome to obtain the SAM file.The `samtosite.py` is used to process SAM file to obtain the site of target A base.This will output a CSV file, which will continue to be processed in the next step.  
```
python samtosite.py -file (.sam)
```
### 1.2 get seq
If you have a known site, you can do this step directly.The sequence with A as the center and the total length of N bp was obtained by using the a site's CSV file and reference genome.This step will get the seq fasta file.  
```
python from_site_to_seq.py -site (.csv) -ref_fa (.fa)
```
## 2. Predict
Use our trained model to predict directly and write the score value of the result into the file which is named by yourself.  
```
python predict.py predict_fa (.fa) -model_path (checkpoint) -outfile (file_name)
```
## 3. Train by yourself
Use our pretreatment process to train a new model with your known A site.We also provide a negative set for your training.More hyperparametric adjustment instructions can be obtained using the `python train.py -h`  
```
python train.py -pos_fa (pos.fa) -neg_fa (neg.fa) -outdir (dir_name)
```
### Additional
If you want to get not only the predicted score, but also the various evaluation indicators of the prediction, you can use `python test.py`.Please refer to `python test.py -h`
