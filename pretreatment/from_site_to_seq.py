import pysam
import re
import pandas as pd
import argparse
from Bio import SeqIO
import argparse
import multiprocessing
from multiprocessing import Process


def A_101bp_seq(wenjian, ref_fa, zongchang):

    def fanxianghubu(s):
        basecomplement = {
            "A": "T",
            "T": "A",
            "G": "C",
            "C": 'G',
            'N': 'N',
        }
        letters = list(s)
        letters = [basecomplement[base] for base in letters][::-1]
        return ''.join(letters)

    center = int((zongchang-1)/2)
    df = pd.read_csv(wenjian, sep='\t')

    for i in range(len(df)):
        chr_location = df.loc[i].tolist()
        chrname = chr_location[0].split('_')[0][3:]
        start = chr_location[1] - center
        end = chr_location[1] + (center + 1)
        ref_start = float(chr_location[0].split('_')[1])
        ref_end = float(chr_location[0].split('_')[2])
        ref = pysam.FastaFile(ref_fa)

        filename = '{}_{}.fasta'.format(wenjian.split('.')[0], zongchang)
        file = open(filename, 'a')
        if start < 0:
            seqx1 = ref.fetch(chrname ,ref_end - (-start), ref_end).upper()
            seqx2 = ref.fetch(chrname ,ref_start, ref_start+end).upper()
            seqx = seqx1 + seqx2
            if seqx[center] == 'A':
                file.write('>' + chr_location[0] + '-' + str(chr_location[1]) + '\n')
                file.write(seqx + '\n')
        elif end > (ref_end - ref_start):
            seqy1 = ref.fetch(chrname ,ref_start+start, ref_end).upper()
            seqy2 = ref.fetch(chrname ,ref_start, ref_start+end-(ref_end-ref_start)).upper()
            seqy = seqy1 + seqy2
            if seqy[center] == 'A':
                file.write('>' + chr_location[0] + '-' + str(chr_location[1]) + '\n')
                file.write(seqy + '\n')
        else:
            seqz = ref.fetch(chrname, ref_start+start, ref_start + end).upper()
            if seqz[center] == 'A':
                file.write('>' + chr_location[0] + '-' + str(chr_location[1]) + '\n')
                file.write(seqz + '\n')

if __name__ == '__main__':
    cpu_n = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser()

    parser.add_argument("-site", "--csv_file", action="store", dest='file', required=True,
                        help=".csv format (from samtosite.py)")
    parser.add_argument("-ref_fa", "--ref_fa", action="store", dest='ref_fa', required=True,
                        help="reference fasta(fa)")

    parser.add_argument("-len", "--len", action="store", dest='len', default=51, type=int,
                        help="seq_len, 以A为center, must be odd")

    args = parser.parse_args()

    sitefile = args.file
    ref_fa = args.ref_fa
    length = args.len

    A_101bp_seq(sitefile, ref_fa ,length)



# if __name__ == '__main__':
#     p = multiprocessing.Pool(24)
#
#     filename = ['./多进程_neg/xae', './多进程_neg/xaq', './多进程_neg/xax', './多进程_neg/xat', './多进程_neg/xal', './多进程_neg/xan', './多进程_neg/xab', './多进程_neg/xau', './多进程_neg/xai', './多进程_neg/xad', './多进程_neg/xaf', './多进程_neg/xar', './多进程_neg/xac', './多进程_neg/xak', './多进程_neg/xam', './多进程_neg/xaa', './多进程_neg/xap', './多进程_neg/xao', './多进程_neg/xaw', './多进程_neg/xas', './多进程_neg/xaj', './多进程_neg/xav', './多进程_neg/xag', './多进程_neg/xah']
#
#
#     for i in filename:
#         #print(pd.read_csv(i, header=None, sep='\t'))
#         # p.apply_async()
#         p.apply_async(A_101bp_seq, args=(i, ref_fa,))
#
#     p.close()
#     p.join()


# A_101bp_seq('./多进程_pos/siteincirc5.csv', ref_fa)
