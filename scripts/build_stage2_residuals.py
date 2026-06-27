import numpy as np
import joblib
import os
import glob
import argparse
from datetime import datetime



class DataNormalizer:
    def __init__(self, stats=None, eps=1e-6):
        """
        stats: dict, 褰㈠ {"mean": np.ndarray(C,), "std": np.ndarray(C,)}
        eps: 闃叉 std 涓?0
        """
        if stats is None:
            self.stats = {"mean": None, "std": None}
        else:
            self.stats = stats
        self.eps = eps

    def fit(self, data):
        """
        鏍规嵁璁粌鏁版嵁鎷熷悎 z-score 鍙傛暟
        data: numpy array, shape = (N, C, H, W)
        """
        if data.ndim != 4:
            raise ValueError(f"data 搴斾负 4 缁?(N, C, H, W)锛屽綋鍓?shape={data.shape}")

        # 姣忎釜閫氶亾鍒嗗埆缁熻 mean/std
        mean = data.mean(axis=(0, 2, 3)).astype(np.float32)   # (C,)
        std = data.std(axis=(0, 2, 3)).astype(np.float32)     # (C,)
        std = np.maximum(std, self.eps)

        self.stats = {
            "mean": mean,
            "std": std
        }

    def transform(self, data):
        """
        鍋?z-score 鏍囧噯鍖?
        鏀寔锛?
            data shape = (N, C, H, W)
            data shape = (C, H, W)
        """
        if self.stats["mean"] is None or self.stats["std"] is None:
            raise RuntimeError("璇峰厛璋冪敤 fit() 鎴?load()")

        mean = self.stats["mean"]
        std = self.stats["std"]

        if data.ndim == 4:
            # (N, C, H, W)
            data_out = (data.astype(np.float32) - mean[None, :, None, None]) / std[None, :, None, None]
        elif data.ndim == 3:
            # (C, H, W)
            data_out = (data.astype(np.float32) - mean[:, None, None]) / std[:, None, None]
        else:
            raise ValueError(f"data 搴斾负 3 缁存垨 4 缁达紝褰撳墠 shape={data.shape}")

        data_out = np.nan_to_num(data_out, nan=0.0, posinf=0.0, neginf=0.0)
        return data_out.astype(np.float32)

    def inverse_transform(self, data):
        """
        鎶?z-score 鏍囧噯鍖栧悗鐨勬暟鎹繕鍘熷埌鍘熷鐗╃悊閲?
        鏀寔锛?
            data shape = (N, C, H, W)
            data shape = (C, H, W)
        """
        if self.stats["mean"] is None or self.stats["std"] is None:
            raise RuntimeError("璇峰厛璋冪敤 fit() 鎴?load()")

        mean = self.stats["mean"]
        std = self.stats["std"]

        if data.ndim == 4:
            data_out = data.astype(np.float32) * std[None, :, None, None] + mean[None, :, None, None]
        elif data.ndim == 3:
            data_out = data.astype(np.float32) * std[:, None, None] + mean[:, None, None]
        else:
            raise ValueError(f"data 搴斾负 3 缁存垨 4 缁达紝褰撳墠 shape={data.shape}")

        return data_out.astype(np.float32)

    def save(self, path):
        """
        淇濆瓨 z-score 鍙傛暟
        """
        joblib.dump(self.stats, path)

    @classmethod
    def load(cls, path, eps=1e-6):
        """
        鍔犺浇 z-score 鍙傛暟
        """
        stats = joblib.load(path)
        return cls(stats=stats, eps=eps)
    

def build_parser():
    parser = argparse.ArgumentParser(
        description="Build SERD stage-2 residual data from stage-1 deterministic corrections."
    )
    parser.add_argument("--data_root_glob", type=str, required=True)
    parser.add_argument("--stage1_prediction_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="./data/stage2_residuals_serd_v1")
    parser.add_argument("--analysis_scaler_path", type=str, default="./data/scalers_ana_zscore_two_step_unet_train.pkl")
    parser.add_argument("--train_count", type=int, default=1292)
    parser.add_argument("--valid_count", type=int, default=92)
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "valid", "test", "all"],
        help="Subset of initialization days to convert using the unified SERD split.",
    )
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    return parser


def select_files(all_paths, split, train_count, valid_count):
    train_end = train_count
    valid_end = train_count + valid_count
    if split == "train":
        return all_paths[:train_end]
    if split == "valid":
        return all_paths[train_end:valid_end]
    if split == "test":
        return all_paths[valid_end:]
    return all_paths


def main():
    args = build_parser().parse_args()
    all_data_path = sorted(glob.glob(args.data_root_glob))
    selected_paths = select_files(all_data_path, args.split, args.train_count, args.valid_count)
    if not selected_paths:
        raise RuntimeError(f"No files selected for split={args.split}")

    normalizer_err = DataNormalizer.load(args.analysis_scaler_path)
    os.makedirs(args.output_root, exist_ok=True)

    saved = 0
    skipped = 0
    for path in selected_paths:
        time0 = os.path.basename(path.rstrip("/\\"))
        time = datetime.strptime(str(time0), "%Y%m%d").strftime("%Y_%m_%d")
        base_time = datetime.strptime(time0, "%Y%m%d").replace(hour=9).strftime("%Y-%m-%d-%H")

        stage1_day_dir = os.path.join(args.stage1_prediction_root, base_time)
        output_day_dir = os.path.join(args.output_root, time0)
        os.makedirs(output_day_dir, exist_ok=True)

        for lead_hour in range(3, 73, 3):
            analysis_path = glob.glob(os.path.join(path, f"{time}__{lead_hour}_analysis.npy"))
            raw_fc_path = glob.glob(os.path.join(path, f"{time}_{lead_hour}.npy"))
            stage1_path = glob.glob(os.path.join(stage1_day_dir, f"{base_time}_{lead_hour:02d}.npy"))
            if len(analysis_path) != 1 or len(raw_fc_path) != 1 or len(stage1_path) != 1:
                skipped += 1
                continue

            analysis = np.load(analysis_path[0])[:, :args.height, :args.width]
            raw_forecast = np.load(raw_fc_path[0])[:, :args.height, :args.width]
            stage1_error = normalizer_err.inverse_transform(np.load(stage1_path[0]))

            if raw_forecast.shape[0] != 45:
                raise ValueError(f"Expected raw forecast to have 45 channels, got {raw_forecast.shape}: {raw_fc_path[0]}")
            corrected_field = raw_forecast.copy()
            corrected_field[::9, :, :] = stage1_error
            residual_error = analysis - stage1_error

            np.save(os.path.join(output_day_dir, f"{time}_{lead_hour}.npy"), corrected_field)
            np.save(os.path.join(output_day_dir, f"{time}_{lead_hour}_err.npy"), residual_error)
            saved += 2

        print(base_time)

    print(f"Finished split={args.split}: saved_files={saved}, skipped_leads={skipped}, output_root={args.output_root}")


if __name__ == "__main__":
    main()


