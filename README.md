# scIDPMs

Source code of "scIDPMs: Single-cell RNA-seq imputation using diffusion probabilistic models".

Single-cell RNA sequencing (scRNA-seq) technology has revolutionized biological research by enabling the sequencing of mRNA in individual cells, thereby providing valuable insights into cellular gene expression and function. However, scRNA-seq data often contains false zero values known as dropout events, which can obscure true gene expression levels and compromise downstream analysis accuracy. To address this issue, several computational approaches have been proposed for imputing missing gene expression values. Nevertheless, these methods face challenges in capturing the distribution of dropout values due to the sparsity of scRNA-seq data and the complexity of gene expression patterns. In this study, we present a novel method called scIDPMs that utilizes conditional diffusion probabilistic models to impute scRNA-seq data. Firstly, scIDPMs identifies dropout sites based on gene expression characteristics and subsequently infers the missing values by considering available gene expression information. To capture global features of gene expression profiles effectively, scIDPMs employs a deep neural network with an attention mechanism to optimize the imputation process. We evaluated the performance of scIDPMs using simulated and real scRNA-seq datasets and compared it with ten other imputation methods. The results demonstrated that scIDPMs outperformed alternative approaches in recovering biologically meaningful gene expression values and improving downstream analyses.



### Running the tests

```python

python exe.py --file_path='./test/counts.csv' --label_path='./test/label.csv'

```



