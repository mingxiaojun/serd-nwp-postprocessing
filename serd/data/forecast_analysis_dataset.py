import numpy as np
import glob
import torch
import os
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta


# --------------------------
# 数据实况时间转换
# --------------------------

def parse_forecast_time(date_str: str, fhour: int) -> tuple[str, str]:
    """
    输入:
        date_str: 形如 '2025_08_02' 的日期字符串（年月日）
        fhour:    预报时效（整数小时，如 6）
    输出:
        (forecast_time_str, init_time_str)，格式 'YYYY-MM-DD-HH'
    """
    # 先按“年月日”解析
    base_time = datetime.strptime(date_str, "%Y_%m_%d").replace(hour=9)

    forecast_time = base_time + timedelta(hours=fhour)
    init_time = base_time

    return forecast_time.strftime("%Y-%m-%d-%H"), init_time.strftime("%Y-%m-%d-%H")

# --------------------------
# 自定义 Dataset
# --------------------------
class ForecastDataset(Dataset):
    def __init__(self, file_paths, normalizer_forecast, normalizer_err):
        self.samples = []
        self.normalizer_forecast = normalizer_forecast
        self.normalizer_err = normalizer_err

        for file_path in file_paths:
            time0 = os.path.basename(file_path.rstrip("/\\"))
            time = datetime.strptime(str(time0), "%Y%m%d").strftime("%Y_%m_%d")
            time = str(time)
            for j in np.arange(3,73,3):
                j = int(j)
                fc = glob.glob(os.path.join(file_path, f"{time}_{j}.npy"))
                fc_err = glob.glob(os.path.join(file_path, f"{time}__{j}_analysis.npy"))
                if len(fc) != 0 and len(fc_err) != 0:
                    parse_time,init0_time = parse_forecast_time(time, j)
                    self.samples.append((fc[0], fc_err[0], j/3,parse_time,init0_time))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fc_path, err_path, lead, turth_time,init1_time = self.samples[idx]
        lead = lead -1

        # 加载 numpy 数据
        fc = np.load(fc_path).reshape(1,45,200,200)
        err = np.load(err_path).reshape(1,5,200,200)

        # 归一化
        # print(fc.shape,err.shape)
        fc = self.normalizer_forecast.transform(fc)
        err = self.normalizer_err.transform(err)

        fc = fc.reshape(45,200,200)
        err = err.reshape(5,200,200)
        # 堆叠
        # data = np.stack([fc, err], axis=0)  # shape [2, ...]
        
        # 转换为 torch.Tensor
        fc = torch.tensor(fc, dtype=torch.float32)
        err = torch.tensor(err, dtype=torch.float32)

        lead = torch.tensor(lead, dtype=torch.long)

        return fc, err, lead, turth_time,init1_time
