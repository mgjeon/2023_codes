import argparse
import glob
import os
import pickle

from tool.metric import evaluate_single
from tool.series_plot import plot_with_flares
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("noaanum", type=int, help="NOAA AR number")
parser.add_argument(
    "--fname",
    type=str,
    help="path to nc file",
    default="/mnt/obsdata/isee_nlfff_v1.2/12673",
)
parser.add_argument("--ext", type=str, help="extension of result file", default="*.nc")
parser.add_argument(
    "--result_dir", type=str, help="path to results.pickle", default="./result"
)
parser.add_argument(
    "--z_Mm", type=int, help="max height for z-direction [Mm]", default=40
)
parser.add_argument("--fig_dir", type=str, help="path to results.pickle", default=".")

args = parser.parse_args()

result_pickle = os.path.join(args.result_dir, "series_result.pickle")

if os.path.exists(result_pickle):
    with open(result_pickle, "rb") as f:
        series_results = pickle.load(f)
        
    if len(series_results['date']) > 1:
        os.makedirs(args.fig_dir, exist_ok=True)
        fig_path = os.path.join(args.fig_dir, f"NOAA_{args.noaanum}.png")
        plot_with_flares(series_results, args.noaanum, fig_path, include_C=False)
        fig_path = os.path.join(args.fig_dir, f"NOAA_{args.noaanum}_C.png")
        plot_with_flares(series_results, args.noaanum, fig_path, include_C=True)
else:
    if os.path.isdir(args.fname):
        files = sorted(glob.glob(os.path.join(args.fname, args.ext)))
    elif os.path.isfile(args.fname):
        files = [args.fname]

    results = []

    for file in tqdm(files):
        results.append(evaluate_single(file, args))
    
    series_results = {k: [r[k] for r in results] for k in results[0].keys()}

    os.makedirs(args.result_dir, exist_ok=True)

    with open(result_pickle, "wb") as f:
        pickle.dump(series_results, f)

    if len(series_results['date']) > 1:
        os.makedirs(args.fig_dir, exist_ok=True)
        fig_path = os.path.join(args.fig_dir, f"NOAA_{args.noaanum}.png")
        plot_with_flares(series_results, args.noaanum, fig_path, include_C=False)
        fig_path = os.path.join(args.fig_dir, f"NOAA_{args.noaanum}_C.png")
        plot_with_flares(series_results, args.noaanum, fig_path, include_C=True)
