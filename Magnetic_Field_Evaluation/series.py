import os 
import glob
import argparse

import pickle

from tool.metric import evaluate_single

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--fname', type=str, help='path to nc file',
                    default='./12673_20170906_000000.nc')
parser.add_argument('--result_dir', type=str, help='path to results.pickle',
                    default='./result')
parser.add_argument('--z_Mm', type=int, help='max height for z-direction [Mm]',
                    default=40)

args = parser.parse_args()

if os.path.isdir(args.fname):
    files = sorted(glob.glob(os.path.join(args.fname, '*.nc')))
elif os.path.isfile(args.fname):
    files = [args.fname]

os.makedirs(args.result_dir, exist_ok=True)

results = []

for file in tqdm(files):
    results.append(evaluate_single(file, args))

results = {k: [r[k] for r in results] for k in results[0].keys()}

result_pickle = os.path.join(args.result_dir, 'result.pickle')
with open(result_pickle, 'wb') as f:
    pickle.dump(results, f)