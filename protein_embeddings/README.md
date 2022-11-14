## Generating Protein Embeddings

**NOTE: You can follow the instructions in `Generate Protein Embeddings.ipynb`**

### Proteome data

Proteomes in Ensembl: https://uswest.ensembl.org/info/data/ftp/index.html

Download protein sequences in FASTA format (below for human and mouse).
```
cd .../proteome
wget http://ftp.ensembl.org/pub/release-105/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz
gunzip Homo_sapiens.GRCh38.pep.all.fa.gz
wget http://ftp.ensembl.org/pub/release-105/fasta/mus_musculus/pep/Mus_musculus.GRCm39.pep.all.fa.gz
gunzip Mus_musculus.GRCm39.pep.all.fa.gz
```


### Generate embeddings from a pretrained Transformer

We can use a pretrained Transformer model from https://github.com/facebookresearch/esm. These models were trained on hundreds of millions of protein sequences from across the tree of life.

**NOTE:** These protein embedding scripts require an older version of the ESM Repo: you should checkout commit:
[`839c5b82c6cd9e18baa7a88dcbed3bd4b6d48e47`](https://github.com/facebookresearch/esm/commit/839c5b82c6cd9e18baa7a88dcbed3bd4b6d48e47)

First, clean the FASTA files to remove the small number of protein sequences with stop codons (and clean up the label of each sequence).
```
python sequence_model/clean_fasta.py \
    --data_path /dfs/project/cross-species/data/proteome/Homo_sapiens.GRCh38.pep.all.fa \
    --save_path /dfs/project/cross-species/data/proteome/Homo_sapiens.GRCh38.pep.all_clean.fa
```

Next, specify the save location for the pretrained model by changing the Torch home location (models are saved in `$TORCH_HOME/hub`).
```
export TORCH_HOME="..." # NOTE: this is a few dozen GB
```

Download the `esm` repo.
```
git clone git@github.com:facebookresearch/esm.git
```

#### ESM-1b Transformer

The ESM-1b Transformer model can generate embeddings from single protein sequences. To generate embeddings using this model, run the following command from the `esm` repo.
```
python extract.py \
    esm1b_t33_650M_UR50S \
    .../proteome/Homo_sapiens.GRCh38.pep.all_clean.fa \
    ../proteome/embeddings/Homo_sapiens.GRCh38.pep.all_clean.fa_esm1b \
    --include mean \
    --truncate
```

This will save the embedding for each protein sequence in a separate `.pt` file in the `.../proteome/embeddings/Homo_sapiens.GRCh38.pep.all_clean.fa_esm1b` directory. Each `.pt` file will contain the mean of the final layer embeddings of each token as the overall sequence embedding.


#### Map gene symbol to protein IDs

Create a mapping from gene symbol to protein IDs to enable embedding lookup.
```
python sequence_model/map_gene_symbol_to_protein_ids.py \
    --fasta_path .../proteome/Homo_sapiens.GRCh38.pep.all.fa \
    --save_path .../proteome/Homo_sapiens.GRCh38.gene_symbol_to_protein_ID.json

python sequence_model/map_gene_symbol_to_protein_ids.py \
    --fasta_path .../proteome/Mus_musculus.GRCm39.pep.all.fa \
    --save_path .../proteome/Mus_musculus.GRCm39.gene_symbol_to_protein_ID.json
```

#### Convert protein embeddings to gene embeddings

```
python sequence_model/convert_protein_embeddings_to_gene_embeddings.py \
    --embedding_dir .../proteome/embeddings/Homo_sapiens.GRCh38.pep.all_clean.fa_esm1b \
    --gene_symbol_to_protein_ids_path .../proteome/Homo_sapiens.GRCh38.gene_symbol_to_protein_ID.json \
    --embedding_model ESM1b \
    --save_path .../proteome/embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM1b.pt

python sequence_model/convert_protein_embeddings_to_gene_embeddings.py \
    --embedding_dir .../proteome/embeddings/Mus_musculus.GRCm39.pep.all_clean.fa_esm1b \
    --gene_symbol_to_protein_ids_path .../proteome/Mus_musculus.GRCm39.gene_symbol_to_protein_ID.json \
    --embedding_model ESM1b \
    --save_path .../proteome/embeddings/Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM1b.pt
```
