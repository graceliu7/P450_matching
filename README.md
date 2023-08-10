# P450_matching
2020 Summer Undergraduate Research Fellowship project on matching cytochrome P450 oxidases and reductases

The work was inspired by this paper: https://www.pnas.org/doi/10.1073/pnas.1901080116#:~:text=We%20now%20report%20that%20genes,phage%20(Mycobacterium%20phage%20Adler). The paper identified several cytochrome P450 type II oxidases in viruses for the first time. Type II P450s, unlike type I, require an additional reductase fragment in order to exhibit catalytic activity, so the scientists weren't able to induce any activity in the discovered P450s. 

My project wanted to identify whether ensemble models and neural networks could be used to match cytochrome P450 oxidase and reductase domains. Since documentation on specific cytochrome oxidase and reductase pairings is incomplete, we decided to try two methods:
1. matching by species - for an unknown oxidase/reductase, find the closest species, and use an existing reductase/oxidase found in that species 
2. matching by cluster - cluster oxidases and reductases by sequence similarity. For an unknown oxidase/reductase, predict the cluster it falls into and choose its counterpart from the corresponding cluster. 

We used the ESM-1b transformer model to represent protein sequences: https://www.pnas.org/doi/10.1073/pnas.2016239118
The reductase_matching folder contains all code for clustering of reductases, and uses several models, including an ensemble model and a neural network for prediction of species and cluster
For the oxidases, due to the size of the dataset (over 300,000 proteins from UniProt), ESM representations could not be stored locally, therefore representation generation and model training had to be done in parallel. This restricted us to using only the neural network, which could be trained in batches, and the code is contained in oxidase_pipeline. 
