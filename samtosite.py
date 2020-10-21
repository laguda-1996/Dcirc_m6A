import pysam
import re
import pandas as pd
import argparse
from Bio import SeqIO
import multiprocessing
from multiprocessing import Process


def find_motif_A(samfile, cpu_n):
    sf = pysam.AlignmentFile(samfile, 'r', threads=cpu_n)
    pos = []
    for s in sf:
        if s.reference_name != None and s.cigarstring == '101M':
            pos_data = []
            query_seq = s.seq

            motif_A_relative_index = [i.start()+2 for i in re.finditer(r'[AGT][AG]AC[AGC]',query_seq)]
            sameSRR_motif_A_refname = s.reference_name
            sameSRR_motif_A_refstart = s.reference_start

            for rel_index in motif_A_relative_index:
                pos_refname = sameSRR_motif_A_refname
                pos_A_location = rel_index + sameSRR_motif_A_refstart
                pos_data = [pos_refname]+ [pos_A_location]
            pos.append(pos_data)


    df = pd.DataFrame(pos)
    df.columns = ['chr', 'location']
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    df.to_csv('{}_motif.csv'.format(samfilename), header=None, index=None, sep='\t')

if __name__ == '__main__':
    cpu_n = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser()

    parser.add_argument("-file", "--file", action="store", dest='file', required=True,
                        help="mapping output(sam format)")

    args = parser.parse_args()

    samfile = args.file

    find_motif_A(samfile, cpu_n)

