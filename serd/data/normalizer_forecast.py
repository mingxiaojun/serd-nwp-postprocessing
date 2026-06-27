import numpy as np
import joblib

class DataNormalizer:
    def __init__(self, stats=None, eps=1e-6):
        """
        stats: dict, 形如 {"mean": np.ndarray(C,), "std": np.ndarray(C,)}
        eps: 防止 std 为 0
        """
        if stats is None:
            self.stats = {"mean": None, "std": None}
        else:
            self.stats = stats
        self.eps = eps

    def fit(self, data):
        """
        根据训练数据拟合 z-score 参数
        data: numpy array, shape = (N, C, H, W)
        """
        if data.ndim != 4:
            raise ValueError(f"data 应为 4 维 (N, C, H, W)，当前 shape={data.shape}")

        # 每个通道分别统计 mean/std
        mean = data.mean(axis=(0, 2, 3)).astype(np.float32)   # (C,)
        std = data.std(axis=(0, 2, 3)).astype(np.float32)     # (C,)
        std = np.maximum(std, self.eps)

        self.stats = {
            "mean": mean,
            "std": std
        }

    def transform(self, data):
        """
        做 z-score 标准化
        支持：
            data shape = (N, C, H, W)
            data shape = (C, H, W)
        """
        if self.stats["mean"] is None or self.stats["std"] is None:
            raise RuntimeError("请先调用 fit() 或 load()")

        mean = self.stats["mean"]
        std = self.stats["std"]

        if data.ndim == 4:
            # (N, C, H, W)
            data_out = (data.astype(np.float32) - mean[None, :, None, None]) / std[None, :, None, None]
        elif data.ndim == 3:
            # (C, H, W)
            data_out = (data.astype(np.float32) - mean[:, None, None]) / std[:, None, None]
        else:
            raise ValueError(f"data 应为 3 维或 4 维，当前 shape={data.shape}")

        data_out = np.nan_to_num(data_out, nan=0.0, posinf=0.0, neginf=0.0)
        return data_out.astype(np.float32)

    def inverse_transform(self, data):
        """
        把 z-score 标准化后的数据还原到原始物理量
        支持：
            data shape = (N, C, H, W)
            data shape = (C, H, W)
        """
        if self.stats["mean"] is None or self.stats["std"] is None:
            raise RuntimeError("请先调用 fit() 或 load()")

        mean = self.stats["mean"]
        std = self.stats["std"]

        if data.ndim == 4:
            data_out = data.astype(np.float32) * std[None, :, None, None] + mean[None, :, None, None]
        elif data.ndim == 3:
            data_out = data.astype(np.float32) * std[:, None, None] + mean[:, None, None]
        else:
            raise ValueError(f"data 应为 3 维或 4 维，当前 shape={data.shape}")

        return data_out.astype(np.float32)

    def save(self, path):
        """
        保存 z-score 参数
        """
        joblib.dump(self.stats, path)

    @classmethod
    def load(cls, path, eps=1e-6):
        """
        加载 z-score 参数
        """
        stats = joblib.load(path)
        return cls(stats=stats, eps=eps)