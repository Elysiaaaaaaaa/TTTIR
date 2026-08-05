#!/usr/bin/env python3
import os
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


def find_latest_run(base_dir):
    runs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    if not runs:
        raise RuntimeError(f'No run directories found under {base_dir}')
    latest = sorted(runs, key=os.path.getmtime, reverse=True)[0]
    return latest


def load_w3_files(run_dir, level):
    pattern = os.path.join(run_dir, '**', f'w3_{level}_*.npy')
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise RuntimeError(f'No w3 files found for level {level} under {run_dir}')
    arrs = []
    names = []
    for f in files:
        try:
            a = np.load(f)
            arrs.append(a.reshape(-1))
            names.append(os.path.basename(f))
        except Exception as e:
            print('Skipping', f, 'due to', e)
    return names, arrs


def compute_pairwise_metrics(arrs):
    N = len(arrs)
    D_rel = np.zeros((N, N), dtype=np.float32)
    D_cos = np.zeros((N, N), dtype=np.float32)
    norms = [np.linalg.norm(a) + 1e-12 for a in arrs]
    for i in range(N):
        ai = arrs[i]
        ni = norms[i]
        for j in range(i, N):
            aj = arrs[j]
            nj = norms[j]
            diff = ai - aj
            dF = np.linalg.norm(diff)
            rel = dF / ((ni + nj) / 2.0)
            # cosine
            cos = float(np.dot(ai, aj) / (ni * nj))
            dcos = 1.0 - cos
            D_rel[i, j] = D_rel[j, i] = rel
            D_cos[i, j] = D_cos[j, i] = dcos
    return D_rel, D_cos


def plot_heatmap(mat, names, out_png, title=''):
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    # do not crowd the tick labels if many
    N = len(names)
    if N <= 30:
        plt.xticks(range(N), names, rotation=90, fontsize=6)
        plt.yticks(range(N), names, fontsize=6)
    else:
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_mds(arrs, out_png, names):
    X = np.vstack(arrs)
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
    coords = mds.fit_transform(X)
    plt.figure(figsize=(6,6))
    plt.scatter(coords[:,0], coords[:,1], s=20)
    for i, n in enumerate(names):
        if len(names) <= 30:
            plt.text(coords[i,0], coords[i,1], n, fontsize=6)
    plt.title('MDS of flattened w3 vectors')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_base_dir', type=str, required=True,
                        help='Base directory used by run_smoke_ttt (e.g. experiments/operator_variation_results)')
    parser.add_argument('--level', type=str, default='conv3',
                        help='Which HSE level to load w3 from (conv1..conv5)')
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    base = args.save_base_dir
    latest = find_latest_run(base)
    print('Using latest run dir:', latest)
    names, arrs = load_w3_files(latest, args.level)
    print(f'Loaded {len(arrs)} w3 files from level {args.level}')

    # Ensure all arrays same length by padding/truncating
    lengths = [a.size for a in arrs]
    L = max(lengths)
    print('Max flattened length:', L)
    arrs_fixed = []
    for a in arrs:
        if a.size < L:
            a2 = np.pad(a, (0, L - a.size), mode='constant')
        else:
            a2 = a[:L]
        arrs_fixed.append(a2)

    D_rel, D_cos = compute_pairwise_metrics(arrs_fixed)

    out_dir = args.out_dir or latest
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'D_rel_{args.level}.npy'), D_rel)
    np.save(os.path.join(out_dir, f'D_cos_{args.level}.npy'), D_cos)

    plot_heatmap(D_rel, names, os.path.join(out_dir, f'heatmap_D_rel_{args.level}.png'), title='Relative Frobenius distance')
    plot_heatmap(D_cos, names, os.path.join(out_dir, f'heatmap_D_cos_{args.level}.png'), title='1 - Cosine similarity')

    plot_mds(arrs_fixed, os.path.join(out_dir, f'mds_{args.level}.png'), names)

    # save pairwise csv
    import csv
    with open(os.path.join(out_dir, f'D_rel_{args.level}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + names)
        for i, n in enumerate(names):
            writer.writerow([n] + list(D_rel[i]))

    print('Saved outputs to', out_dir)

if __name__ == '__main__':
    main()
