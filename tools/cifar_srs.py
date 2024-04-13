import argparse
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def create_srs(input_folder, output_folder, n):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)
    if "labels.npy" not in files:
        print("No labels archive found, exiting...")
        return
    
    # labels[x] -> label for image x
    labels = np.load(os.path.join(input_folder, "labels.npy"))

    # now load all archives
    archives = {
        file: np.load(os.path.join(input_folder, file)) 
        for file in files if file != "labels.npy" and file.endswith('.npy')
    }

    srs_sampler = StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=7)
   
    _, test_indices = list(srs_sampler.split(list(archives.values())[0], labels))[0]
    # use test as that's size `n`
    
    for file, archive in archives.items():
        new_filename = f"{file.rstrip('.npy')}_srs{n}.npy"
        file_path = os.path.join(output_folder, new_filename)
        print(file_path)
        np.save(file_path, archive[test_indices])

    # save subset of labels
    label_subset_path = os.path.join(output_folder, f"labels_srs{n}.npy")
    np.save(label_subset_path, labels[test_indices])
    print(label_subset_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files from one folder to another.")
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument("sample", help="Subsample size to create", type=int)

    args = parser.parse_args()
    create_srs(args.input_folder, args.output_folder, args.sample)
