import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

def split_parquet_to_individual_hex(input_path, output_folder, bit_depth=8, max_samples=100):
    try:
        # Read the 'audio' column
        table = pq.read_table(input_path, columns=['audio','label'])
        audio_column = table.column('audio').to_pylist()
        label_column = table.column('label').to_pylist()
        
        for i, row in enumerate(audio_column):
            if i >= max_samples:
                break
            label = label_column[i]    
            raw_bytes = row['bytes']
            # Convert binary to 16-bit PCM
            data = np.frombuffer(raw_bytes, dtype=np.int16)

            # 1. Downsample and Slice
            data = data[::4]       
            data = data[:16384]    
            
            # 2. Robust Normalization
            if len(data) > 0:
                data = data.astype(np.float32)
                # Define min/max clearly
                d_min = np.min(data)
                d_max = np.max(data)
                
                if d_max != d_min:
                    # Scale to 0 - 255
                    data = (data - d_min) / (d_max - d_min) * (2**bit_depth - 1)
                else:
                    # If the sample is total silence, just set to middle-ground (128)
                    data = np.full_like(data, 128)
                
                data = data.astype(np.uint8)

                # 3. Save to hex
                file_name = f"{input_path.stem}_s{i:03d}_L{label}.hex"
                with open(output_folder / file_name, 'w') as f:
                    for val in data:
                        f.write(f"{val:02x}\n")
            
        print(f"Finished {input_path.name}: Generated individual hex files.")

    except Exception as e:
        print(f"Error on {input_path.name}: {e}")

# --- EXECUTION ---
base_dir = Path(r"C:\Users\User\OneDrive\Documents\EEE405project")
export_dir = base_dir / "individual_samples"
export_dir.mkdir(exist_ok=True)

for p_file in base_dir.glob("*.parquet"):
    split_parquet_to_individual_hex(p_file, export_dir, max_samples=4500) # Start with 5 to test