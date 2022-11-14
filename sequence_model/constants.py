"""Contains constants used for sequences and sequence models."""
# Header for BLAST output file using outfmt 6 (https://www.metagenomics.wiki/tools/blast/blastn-output-format-6)
BLAST_HEADER = [
    'qseqid',
    'sseqid',
    'pident',
    'length',
    'mismatch',
    'gapopen',
    'qstart',
    'qend',
    'sstart',
    'send',
    'evalue',
    'bitscore'
]

# Vocabulary of amino acid characters
# Note: 0 reserved for pad token; '-' for class token where full sequence embedding will be extracted
# Note: '*' is for stop codon; 'X' is for any amino acid; 'U' is for Selenocysteine
VOCAB = {
    character: index
    for index, character in enumerate([
        '-', '*', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
    ])
}

# Last layer of pretrained transformer
LAST_LAYER = 33 # ESM1b
MSA_LAST_LAYER = 12 # MSA
LAST_LAYER_2 = 48 # ESM2