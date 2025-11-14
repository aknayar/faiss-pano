# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, Generator, List, Optional, Tuple
from faiss.contrib.datasets import Dataset



class CSVDataset(Dataset):
    """Custom dataset class for loading data from CSV files and fvecs files."""
    
    def __init__(self, csv_path: str, nq: int, nb: Optional[int] = None, seed: int = 1338, query_indices: Optional[np.ndarray] = None, test_csv_path: Optional[str] = None):
        """
        Initialize dataset from CSV or fvecs file.
        
        Args:
            csv_path: Path to the CSV or fvecs file (training data)
            nq: Number of query points to sample (only used if test_csv_path is None)
            nb: Maximum number of database points to load (None = load all)
            seed: Random seed for reproducible sampling
            query_indices: Predetermined query indices to use (if None, will sample randomly)
            test_csv_path: Path to separate test file for queries (if provided, uses all points from this file as queries)
        """
        self.file_path = csv_path  # Can be CSV or fvecs
        self.test_file_path = test_csv_path
        self.nq = nq
        self.nb = nb
        self.seed = seed
        self.query_indices = query_indices
        self._load_data(read_base=True)
    
    def _read_fvecs(self, filename: str) -> np.ndarray:
        """Read fvecs format file."""
        with open(filename, 'rb') as f:
            vectors = []
            while True:
                # Read dimension (4 bytes, little endian)
                dim_bytes = f.read(4)
                if len(dim_bytes) != 4:
                    break
                dim = int.from_bytes(dim_bytes, byteorder='little')
                
                # Read the vector (dim * 4 bytes for floats)
                vector_bytes = f.read(dim * 4)
                if len(vector_bytes) != dim * 4:
                    break
                
                # Convert bytes to float32 array
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                vectors.append(vector)
                
                # Apply limit if specified
                if self.nb is not None and len(vectors) >= self.nb:
                    break
            
            return np.array(vectors, dtype=np.float32)

    def _load_data(self, read_base=True):
        """Load and preprocess data from CSV or fvecs files."""
        
        # Load training data (database)
        if read_base:
            print(f"Loading training data from {self.file_path}...")
            self.data = self._load_file_data(self.file_path, self.nb, "training")
        else:
            self.data = None
        
        # Load query data from separate test file if provided
        if self.test_file_path is not None:
            print(f"Loading test data from {self.test_file_path}...")
            self.query_data = self._load_file_data(self.test_file_path, self.nq, "test")
            print(f"Using all {len(self.query_data)} points from test file as queries")
            # No need for query_indices since we use all test data
            self.query_indices = None
        else:
            # Use predetermined query indices or sample new ones from training data
            if self.query_indices is not None:
                # Use provided query indices
                print(f"Using predetermined query indices: {len(self.query_indices)} points")
                # Validate indices are within bounds
                if np.max(self.query_indices) >= len(self.data):
                    raise ValueError(f"Query indices contain values >= dataset size ({len(self.data)})")
            else:
                # Set random seed for reproducible sampling
                np.random.seed(self.seed)
                
                # Sample query points
                if self.nq >= len(self.data):
                    print(f"Warning: nq ({self.nq}) >= dataset size ({len(self.data)}), using all data as queries")
                    self.query_indices = np.arange(len(self.data))
                else:
                    self.query_indices = np.random.choice(len(self.data), size=self.nq, replace=False)
                
                print(f"Selected {len(self.query_indices)} query points")
            
            # No separate query data, will use indices into training data
            self.query_data = None
    
    def _load_file_data(self, file_path: str, row_limit: Optional[int], data_type: str) -> np.ndarray:
        """Load data from a single file (CSV or fvecs)."""
        # Determine file type based on extension
        if file_path.lower().endswith('.fvec') or file_path.lower().endswith('.fvecs'):
            print(f"Detected fvecs file format for {data_type} data")
            # Temporarily store the row limit and restore it
            original_nb = self.nb
            self.nb = row_limit
            data = self._read_fvecs(file_path)
            self.nb = original_nb
            print(f"Loaded {data_type} fvecs with shape: {data.shape}")
        else:
            print("NON FVECS/FVEC FILE!!!")
            exit(1)
        return data
    
    def get_database(self) -> np.ndarray:
        """Return the full dataset as database vectors."""
        return self.data
    
    def get_queries(self) -> np.ndarray:
        """Return query vectors (either from separate test file or sampled from training data)."""
        if self.query_data is not None:
            # Use all data from separate test file
            return self.query_data
        else:
            # Use sampled points from training data
            return self.data[self.query_indices]
    
    def get_groundtruth(self, k: int = 10) -> np.ndarray:
        """Compute ground truth using brute force search."""
        print("Computing ground truth with brute force...")
        index = faiss.IndexFlatL2(self.data.shape[1])
        index.add(self.data)
        _, I = index.search(self.get_queries(), k)
        return I


ds = CSVDataset(csv_path="../blessing/sift100m_train_Cayley_Transform_20250920_013830_10.fvecs", test_csv_path="../blessing/sift100m_test_Cayley_Transform_20250920_013830_10.fvecs", nq=1, nb=100000)


nq = 1
nb = 100000
xq = ds.get_queries()[:nq]
xb = ds.get_database()[:nb]
# gt = ds.get_groundtruth()[:nq]

# xt = ds.get_train()

nb, d = xb.shape
# nt, d = xt.shape

k = 10
# gt = gt[:, :k]


def eval_qps(index):
    faiss.cvar.indexPanorama_stats.reset()
    t0 = time.time()
    _, I = index.search(xq, k=k)
    t = time.time() - t0
    speed = t * 1000 / nq  # ms/query
    qps = 1000 / speed

    # corrects = (gt == I).sum()
    # recall = corrects / (nq * k)
    recall = 1.0
    ratio_dims_scanned = faiss.cvar.indexPanorama_stats.ratio_dims_scanned
    print(
        f"\tRecall@{k}: {recall:.6f}, speed: {speed:.6f} ms/query, "
        f"dims scanned: {ratio_dims_scanned * 100:.2f}%"
    )
    return recall, qps


def build_index(name):
    index = faiss.index_factory(d, name)

    faiss.omp_set_num_threads(mp.cpu_count())
    index.train(xb)
    index.add(xb)

    faiss.omp_set_num_threads(1)
    return index


nlevels = 16
batch_size = 512

plt.figure(figsize=(8, 6), dpi=80)

names = [
    # "Flat",
    f"PCA{d},FlatL2Panorama{nlevels}_{batch_size}",
]

labels = []
qps_values = []

for name in names:
    print(f"======{name}")
    index = build_index(name)
    recall, qps = eval_qps(index)
    labels.append(f"{name}\n(r@{recall:.3f})")
    qps_values.append(qps)

# x = np.arange(len(names))
# plt.bar(x, qps_values, color=['#1f77b4', '#ff7f0e'])
# speedup = qps_values[1] / qps_values[0]
# ax = plt.gca()
# ax.text(
# 	x[1],
# 	qps_values[1] * 1.01,
# 	f"{speedup:.2f}x",
# 	ha="center",
# 	va="bottom",
# )
# plt.xticks(x, labels, rotation=0)
# plt.ylabel("QPS")
# plt.title("Flat Indexes on GIST1M")

# plt.tight_layout()
# plt.savefig("bench_flat_l2_panorama.png", bbox_inches="tight")
