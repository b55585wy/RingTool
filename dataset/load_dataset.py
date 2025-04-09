from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def load_dataset(config: Dict, raw_data: Dict, task: str="hr"):
    if task not in ["hr", "bp", "rr", "spo2"]:
        raise ValueError("Invalid task. Choose 'hr', 'bp', 'rr', or 'spo2'.")
    if task == "hr":
        return HeartrateDataset(config, raw_data)
    # randomize one sample with the shape of (77,30)
    return [(torch.randn(201, 1), torch.randn(1))]



class HeartrateDataset(Dataset):
    def __init__(self, config: Dict, raw_data: List):
        self.raw_data = raw_data
        self.config = config
        self.load_data()
        # import ipdb; ipdb.set_trace()
    
        
    def load_data(self):
        self.data = []
        window_size = 10 # second
        for participant in tqdm(self.raw_data):
            if len(participant['ring1']) == 0 or len(participant['hr']) == 0:
                continue
            ring1_timestamps = participant['ring1']['timestamp'].tolist()
            ring1_irs = participant['ring1']['ir'].tolist()
            hr_windows = [
                (timestamp - window_size // 2, timestamp + window_size // 2)
                for timestamp in participant['hr']['timestamp']
            ]
            ring1_idx = 0
            for idx, gt in participant['hr'].iterrows():
                timestamp = gt['timestamp']
                hr = gt['hr']
                start_t, end_t = hr_windows[idx]
                # Move the ring1_idx to the start of the window
                while ring1_idx < len(ring1_timestamps) and ring1_timestamps[ring1_idx] < start_t:
                    ring1_idx += 1
                # Collect the ir values within the window
                ir_window = []
                while ring1_idx < len(ring1_timestamps) and ring1_timestamps[ring1_idx] <= end_t:
                    ir_window.append(ring1_irs[ring1_idx])
                    ring1_idx += 1
                # Reset ring1_idx for the next window
                ring1_idx -= len(ir_window)
                if len(ir_window) < 990:
                    continue
                # Pad or truncate the ir_window to 1000 samples
                if len(ir_window) > 1000:
                    ir_window = ir_window[:1000]
                else:
                    ir_window += [0] * (1000 - len(ir_window))
                hr_tensor = torch.tensor(hr, dtype=torch.float32)
                ir_tensor = torch.tensor(ir_window, dtype=torch.float32)
                ir_tensor = ir_tensor.unsqueeze(-1)
                self.data.append((ir_tensor, hr_tensor))
            # for idx, gt in participant['hr'].iterrows():
                # timestamp = gt['timestamp']
                # hr = gt['hr']
                # start_t = timestamp - window_size // 2
                # end_t = timestamp + window_size // 2
                # window_data = participant['ring1'][(participant['ring1'] >= start_t) & (participant['ring1'] <= end_t)]
                # ir = window_data['ir'].tolist()
                # hr = torch.tensor(hr, dtype=torch.float32)
                # ir = torch.tensor(ir, dtype=torch.float32)
                # self.data.append((ir, hr))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming each item in the dataset is a tuple (input, label)
        return self.data[idx]
