"""Script to map gene symbol to protein IDs based on Ensembl FASTA files."""
import json
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    fasta_path: Path  # Path to FASTA protein sequence file from Ensembl.
    save_path: Path  # Path to a .json file where the mapping from gene symbol to protein IDs will be saved.

    def process_args(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)


GENE_SYMBOL = 'gene_symbol:'
# GENE_SYMBOL = 'gene:' # for octopus



def map_gene_symbol_to_protein_ids(args: Args) -> None:
    """Script to map gene symbol to protein IDs based on Ensembl FASTA files."""
    # Load sequences
    seqs = list(SeqIO.parse(args.fasta_path, 'fasta'))

    # Map from gene symbol to protein IDs
    gene_symbol_to_protein_ids = defaultdict(set)

    for seq in tqdm(seqs):
        description = seq.description.split()
        gene_symbols = [text for text in description if text.startswith(GENE_SYMBOL)]

        if len(gene_symbols) > 1:
            raise ValueError('Sequence can have at most one gene symbol.')

        if len(gene_symbols) == 1:
            gene_symbol = gene_symbols[0][len(GENE_SYMBOL):]
            gene_symbol_to_protein_ids[gene_symbol].add(seq.id)

    # Print stats
    all_gene_symbols = set(gene_symbol_to_protein_ids)
    print(len(seqs))
    all_protein_ids = set.union(*[protein_ids for protein_ids in gene_symbol_to_protein_ids.values()])

    print(f'Number of gene symbols = {len(all_gene_symbols):,}')
    print(f'Number of protein IDs = {len(all_protein_ids):,}')

    # Convert object to JSON-acceptable types
    gene_symbol_to_protein_ids = {
        gene_symbol: sorted(protein_ids)
        for gene_symbol, protein_ids in gene_symbol_to_protein_ids.items()
    }

    # Save mapping
    with open(args.save_path, 'w') as f:
        json.dump(gene_symbol_to_protein_ids, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    map_gene_symbol_to_protein_ids(Args().parse_args())