import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42) # Setting global seed (pandas and scikit-learn both default to numpy seed) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel-data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--test-proportion', type=float, default=.2)
    return parser.parse_args()

args = parse_args()

parallel_data = Path(args.parallel_data_dir)

dfs = []
for x in parallel_data.iterdir():
    if x.is_dir():
        source, target = x.name.split('2')
        for y in x.iterdir():
            if y.is_file():
                title = ''.join(y.stem.split('_')[:-1]) # Title of the corpus
                df = pd.read_csv(y, sep='\|\|\|', names=['source_text','target_text'], encoding='utf-8', engine='python', quotechar=None, quoting=3, doublequote=False)
                df['source_lang'] = source
                df['target_lang'] = target
                df['source_title'] = title
                df['target_title'] = title
                dfs.append(df) # Collect all the corpora into a single data frame

pairs = pd.concat(dfs, 0, sort=False) # This data frame contains only positive pairs
pairs = pairs[(pairs.source_lang!='nl') & (pairs.target_lang!='nl')] # Remove Dutch as we are not interested in it
pairs = pairs.sample(frac=1) # Give it a good shuffle

# Split between train and test sets for positive pairs only (we don't care about performing well on negative pairs). We stratify by language pairs and corpus title
pairs_train, pairs_test = train_test_split(pairs, train_size=1-args.test_proportion, stratify=pairs.apply(lambda row: '~'.join([row['source_lang'],row['target_lang'],row['source_title'],row['target_title']]),1))

out_dir = Path(args.output_dir)
pairs_train.to_csv(out_dir/'train.tsv', columns=['source_text','source_lang','source_title','target_text','target_lang','target_title'], index=False, header=False, sep='\t')
pairs_test.to_csv(out_dir/'test.tsv', columns=['source_text','source_lang','source_title','target_text','target_lang','target_title'], index=False, header=False, sep='\t')