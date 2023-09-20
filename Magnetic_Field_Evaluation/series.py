import argparse
import glob
import os
import pickle

from tqdm import tqdm

from tool.metric import evaluate_single

parser = argparse.ArgumentParser()

parser.add_argument(
    "--fname",
    type=str,
    help="path to nc file",
    default="/mnt/obsdata/isee_nlfff_v1.2/12673",
)
parser.add_argument(
    "--result_dir", type=str, help="path to results.pickle", default="./result"
)
parser.add_argument(
    "--z_Mm", type=int, help="max height for z-direction [Mm]", default=40
)

args = parser.parse_args()

if os.path.isdir(args.fname):
    files = sorted(glob.glob(os.path.join(args.fname, "*.nc")))
elif os.path.isfile(args.fname):
    files = [args.fname]

os.makedirs(args.result_dir, exist_ok=True)

results = []

for file in tqdm(files):
    results.append(evaluate_single(file, args))

results = {k: [r[k] for r in results] for k in results[0].keys()}

result_pickle = os.path.join(args.result_dir, "series_result.pickle")
with open(result_pickle, "wb") as f:
    pickle.dump(results, f)
