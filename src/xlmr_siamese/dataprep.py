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

pos_pairs = pd.concat(dfs, 0, sort=False) # This data frame contains only positive pairs
pos_pairs = pos_pairs[(pos_pairs.source_lang!='nl') & (pos_pairs.target_lang!='nl')] # Remove Dutch as we are not interested in it
pos_pairs['pair_type'] = 'pos'
pos_pairs = pos_pairs.sample(frac=1) # Give it a good shuffle

# To create negative pairs, we first stack source and target translations into a single data frame, losing the pairing
sources = pos_pairs[['source_text','source_lang','source_title']].rename(columns={'source_text':'text', 'source_lang':'lang', 'source_title':'title'})
targets = pos_pairs[['target_text','target_lang','target_title']].rename(columns={'target_text':'text', 'target_lang':'lang', 'target_title':'title'})
unpaired = pd.concat([sources,targets], 0).drop_duplicates()

# Now we take the cartesian product of all languages and randomly pair sentences from one language to the other (including the same language)
neg_pairs = []
for lang1,g1 in unpaired.groupby('lang'):
    for lang2,g2 in unpaired.groupby('lang'):
        g1 = g1.sample(frac=1).rename(columns={'text':'source_text','lang':'source_lang','title':'source_title'}).reset_index(drop=True)
        g2 = g2.sample(frac=1).rename(columns={'text':'target_text','lang':'target_lang','title':'target_title'}).reset_index(drop=True)
        neg_pairs.append(pd.concat([g1,g2], 1, join='inner')) # 'inner' here simply means if there are less sentences in lang1 than in lang2, then we only pair them up to the size of lang1

neg_pairs = pd.concat(neg_pairs, 0, sort=False)
neg_pairs['pair_type'] = 'neg'

# Split between train and test sets for positive pairs only (we don't care about performing well on negative pairs). We stratify by language pairs and corpus title
pos_pairs_train, pos_pairs_test = train_test_split(pos_pairs, train_size=1-args.test_proportion, stratify=pos_pairs.apply(lambda row: '~'.join([row['source_lang'],row['target_lang'],row['source_title'],row['target_title']]),1))

# Give it one last good shuffle
train = pd.concat([pos_pairs_train, neg_pairs], 0, sort=False).sample(frac=1)
test = pos_pairs_test.sample(frac=1)

out_dir = Path(args.output_dir)
train.to_csv(out_dir/'train.tsv', columns=['source_text','source_lang','source_title','target_text','target_lang','target_title','pair_type'], index=False, header=False, sep='\t')
test.to_csv(out_dir/'test.tsv', columns=['source_text','source_lang','source_title','target_text','target_lang','target_title','pair_type'], index=False, header=False, sep='\t')