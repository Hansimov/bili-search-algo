import numpy as np

from pathlib import Path
from tclogger import logger


class LSHConverter:
    """Convert emb-floats to hash-bits with LSH.
    - dims: input embedding floats dimension (1024)
    - bitn: output hash bits num (2048)
    - seed: random seed for reproducibility
    """

    def __init__(self, dims: int = 1024, bitn: int = 2048, seed: int = 42):
        self.dims = dims
        self.bitn = bitn
        self.seed = seed
        self.init_hyperplanes()

    def init_hyperplanes(self):
        """init random hyper-planes matrix"""
        self.hps_path = Path(__file__).parent / f"lsh_hps_{self.dims}d_{self.bitn}b.npy"
        if self.hps_path.exists():
            self.load_hyperplanes()
        else:
            self.generate_hyperplanes()
            self.save_hyperplanes()

    def generate_hyperplanes(self):
        np.random.seed(self.seed)
        # generate random hyperplanes: (bitn * dims)
        # each row is a random hyperplane normal vector with dims elements
        self.hps = np.random.randn(self.bitn, self.dims).astype(np.float32)
        # normalize hyperplanes
        self.hps = self.hps / np.linalg.norm(self.hps, axis=1, keepdims=True)

    def embs_to_bits(self, embs: np.ndarray) -> np.ndarray:
        """convert float-embs to hash-bits (ndarray)

        Input:
        - embs: with shape (n, dims) or (dims,)
        - n: number of samples/rows

        Output:
        - bits: with shape (n, bitn) or (bitn,)
        """
        # reshape for single row
        if embs.ndims == 1:
            embs = embs.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        # project embs onto hyperplanes: (n, bits)
        projs = np.dot(embs, self.hps.T)
        # >0 maps to 1, <=0 maps to 0
        bits = (projs > 0).astype(np.uint8)
        if squeeze:
            return bits[0]
        else:
            return bits

    def bits_to_hex(self, bits: np.ndarray) -> str:
        """convert hash bits to hex str.

        Input:
        - bits: with shape (bits,)

        Output:
        - hex_str: hex str with length len(bits)/4
        """
        # pad to 8x
        bitn = len(bits)
        n_bytes = (bitn + 7) // 8
        padded = np.zeros(n_bytes * 8, dtype=np.uint8)
        padded[:bitn] = bits
        # pack bits into bytes
        bytes_arr = np.packbits(padded)
        hex_str = bytes_arr.tobytes().hex()
        return hex_str

    def save_hyperplanes(self):
        logger.note(f"> Save LSH HyperPlanes to:")
        np.save(self.hps_path, self.hps)
        logger.okay(f"  * {self.hps_path}")

    def load_hyperplanes(self):
        logger.note(f"> Load LSH HyperPlanes from:")
        self.hps = np.load(self.hps_path)
        self.bitn, self.dims = self.hps.shape
        logger.file(f"  * {self.hps_path}")
