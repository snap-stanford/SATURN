"""Convert protein embeddings to gene embeddings by averaging the protein embeddings for each gene."""
import json
from pathlib import Path
from typing_extensions import Literal

import torch
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    embedding_dir: Path  # Path to directory containing .pt files with protein embeddings labeled by protein ID.
    gene_symbol_to_protein_ids_path: Path  # Path to .json file containing a mapping from gene symbol to protein IDs.
    embedding_model: Literal['ESM1b', 'MSA1b', 'ESM2']  # Model used to generate the protein embeddings.
    save_path: Path  # Path to .pt file where a map from gene symbol to embedding will be saved.

    def process_args(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)


from constants import LAST_LAYER, MSA_LAST_LAYER, LAST_LAYER_2


def convert_protein_embeddings_to_gene_embeddings(args: Args) -> None:
    """Convert protein embeddings to gene embeddings by averaging the protein embeddings for each gene."""
    # Load gene symbol to protein IDs map
    with open(args.gene_symbol_to_protein_ids_path) as f:
        gene_symbol_to_protein_ids = json.load(f)

    # Get union of protein IDs
    all_protein_ids = set.union(*[
        set(protein_ids) for protein_ids in gene_symbol_to_protein_ids.values()
    ])

    # Get protein embedding paths
    protein_embedding_paths = [args.embedding_dir / f'{protein_id}.pt' for protein_id in all_protein_ids]

    # Get last layer
    if args.embedding_model == 'ESM1b':
        last_layer = LAST_LAYER
    elif args.embedding_model == 'MSA1b':
        last_layer = MSA_LAST_LAYER
    elif args.embedding_model == 'ESM2':
        last_layer = LAST_LAYER_2
    else:
        raise ValueError(f'Embedding model "{args.embedding_model}" is not supported.')

    # Load mapping from protein ID to embedding
    protein_id_to_embedding = {
        protein_embedding_path.stem: torch.load(protein_embedding_path)['mean_representations'][last_layer]
        for protein_embedding_path in tqdm(protein_embedding_paths)
        if protein_embedding_path.exists()
    }
    print(protein_embedding_paths[0])

    # Update gene symbol to protein ID mapping based on available embeddings
    available_protein_ids = set(protein_id_to_embedding)

    gene_symbol_to_protein_ids = {
        gene_symbol: set(protein_ids) & available_protein_ids
        for gene_symbol, protein_ids in gene_symbol_to_protein_ids.items()
    }

    gene_symbol_to_protein_ids = {
        gene_symbol: protein_ids
        for gene_symbol, protein_ids in gene_symbol_to_protein_ids.items()
        if len(protein_ids) > 0
    }

    # Average protein embeddings for each gene to get gene embeddings
    gene_symbol_to_embedding = {
        gene_symbol: torch.mean(
            torch.stack([
                protein_id_to_embedding[protein_id] for protein_id in protein_ids
            ]),
            dim=0
        )
        for gene_symbol, protein_ids in tqdm(gene_symbol_to_protein_ids.items())
    }

    # Save gene symbol to embedding map
    torch.save(gene_symbol_to_embedding, args.save_path)


if __name__ == '__main__':
    convert_protein_embeddings_to_gene_embeddings(Args().parse_args())
