import torch
from torch.utils.data import Dataset
import numpy as np
import json

class EHRDataset(Dataset):
    def __init__(self, path_documents="events.json", path_labels="targets.json", path_tokenizer="tokenizer.json", mode="train", sequence_length=100):
        assert mode in ["train", "test"]
        self.sequence_length = sequence_length  # Define the maximum sequence length

        # Load data1
        with open(path_documents) as f:
            self.data = json.load(f)
        with open(path_labels) as f:
            self.targets = json.load(f)
        with open(path_tokenizer) as f:
            self.tokenizer = json.load(f)

        # Split data into training and testing sets based on key endings
        ref_k = list(self.data.keys()).copy()
        if mode == "train":
            for k in ref_k:
                if k.endswith("8") or k.endswith("9"):
                    del self.data[k]
                    del self.targets[k]
        else:  # Test mode
            for k in ref_k:
                if not k.endswith("8") and not k.endswith("9"):
                    del self.data[k]
                    del self.targets[k]

        self.icu_stays_id = list(self.data.keys())  # Store patient IDs

        assert len(self.data) == len(self.targets)

    def __len__(self):
        return len(self.icu_stays_id)

    def __getitem__(self, index):
        # Load patient data
        patient = self.data[self.icu_stays_id[index]]
        t_list, v_list = list(map(float, patient.keys())) + [179.], list(patient.values()) + [[[1, 0]]]  # Add CLS token

        # Expand timestamps to match the event count
        minutes = np.repeat(t_list, list(map(len, v_list)))
        minutes = torch.tensor(minutes).long()

        # Encode codes and extract values
        codes = torch.tensor([self.tokenizer.get(str(e[0]), len(self.tokenizer)) for v in v_list for e in v]).long()
        values = torch.tensor([e[1] for v in v_list for e in v])

        seq_l = minutes.size(0)  # Actual sequence length before padding/truncation

        # Truncate if sequence is longer than the maximum length
        if minutes.size(0) > self.sequence_length:
            minutes = minutes[-self.sequence_length:]
            codes = codes[-self.sequence_length:]
            values = values[-self.sequence_length:]

        # Padding if the sequence is shorter than the maximum length
        padding_length = self.sequence_length - minutes.size(0)
        if padding_length > 0:
            minutes = torch.nn.functional.pad(minutes, (padding_length, 0))
            codes = torch.nn.functional.pad(codes, (padding_length, 0))
            values = torch.nn.functional.pad(values, (padding_length, 0))

        # Create attention mask (1 for valid tokens, 0 for padding)
        attention_mask = torch.zeros(self.sequence_length, dtype=torch.long)
        attention_mask[-seq_l:] = 1  # Valid tokens at the end, padding at the front

        # Prepare the sample dictionary
        sample = {
            "codes": codes,                  # Encoded medical event codes
            "values": values,                # Corresponding values for each code
            "minutes": minutes,              # Timestamps of events
            "attention_mask": attention_mask,  # Attention mask for the model
            "target": 1 - self.targets[self.icu_stays_id[index]],  # Target label (inverted)
            "seq_l": seq_l                   # Original sequence length (before padding)
        }

        return sample
