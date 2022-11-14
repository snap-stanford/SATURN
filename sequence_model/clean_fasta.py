"""Script to clean FASTA sequences (e.g., remove sequences with stop codons) to use with ESM (https://github.com/facebookresearch/esm)."""
from pathlib import Path

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    data_path: Path  # Path to FASTA file to clean.
    save_path: Path  # Path where cleaned FASTA file will be saved.

    def process_args(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)


def clean_fasta(args: Args) -> None:
    """Cleans FASTA sequences (e.g., remove sequences with stop codons)."""
    # Load sequences
    seqs = list(SeqIO.parse(args.data_path, 'fasta'))
    print(f'Number of original sequences = {len(seqs):,}')

    # Clean sequences
    seqs = [SeqRecord(seq=seq.seq, id=seq.id, description='') for seq in tqdm(seqs) if '*' not in seq.seq]
    print(f'Number of cleaned sequences = {len(seqs):,}')

    # Save sequences
    SeqIO.write(seqs, args.save_path, 'fasta')


if __name__ == '__main__':
    clean_fasta(Args().parse_args())
