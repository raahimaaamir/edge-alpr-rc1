import pandas as pd
import numpy as np
from pathlib import Path

# Load master manifest
manifest_path = Path("C:/Projects/alpr/data/manifests/manifest_master.csv")
df = pd.read_csv(manifest_path)

# Extract unique track IDs per dataset
ufpr_tracks = df[df['dataset'] == 'ufpr']['track_id'].unique()
rodosol_tracks = df[df['dataset'] == 'rodosol']['track_id'].unique()

np.random.seed(42)  # Fixed seed for reproducibility
np.random.shuffle(ufpr_tracks)
np.random.shuffle(rodosol_tracks)

# 80/20 train/val split on unique tracks
ufpr_split = int(len(ufpr_tracks) * 0.8)
rodosol_split = int(len(rodosol_tracks) * 0.8)

train_tracks = set(ufpr_tracks[:ufpr_split]).union(set(rodosol_tracks[:rodosol_split]))
val_tracks = set(ufpr_tracks[ufpr_split:]).union(set(rodosol_tracks[rodosol_split:]))

train_df = df[df['track_id'].isin(train_tracks)].reset_index(drop=True)
val_df = df[df['track_id'].isin(val_tracks)].reset_index(drop=True)

train_df.to_csv("C:/Projects/alpr/data/manifests/manifest_train.csv", index=False)
val_df.to_csv("C:/Projects/alpr/data/manifests/manifest_val.csv", index=False)

print(f"Train samples: {len(train_df)} across {len(train_tracks)} tracks")
print(f"Validation samples: {len(val_df)} across {len(val_tracks)} tracks")