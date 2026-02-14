import shutil
import os
import argparse

def main(args):
    source_folder = args.source_folder
    destination_folder = args.destination_folder
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    destination_folder = os.listdir(destination_folder)
    shift = len(destination_folder)
    
    source_folder = os.listdir(source_folder)
    for file in source_folder:
        if file.endswith(".zip"):
            file_number = int(file.split("_")[1].split(".")[0])
            new_file_number = file_number + shift
            new_file_name = f"model_{new_file_number}.zip"
            shutil.copyfile(os.path.join(args.source_folder, file), os.path.join(args.destination_folder, new_file_name))

    new_total_of_enemies = len(source_folder) + shift

    print(f"Total of enemies after migration: {new_total_of_enemies}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate pool files to new structure.")
    parser.add_argument("--source_folder", type=str, help="Path to the source folder containing pool files.")
    parser.add_argument("--destination_folder", type=str, help="Path to the destination folder for migrated pool files.", default="enemy_model")
    args = parser.parse_args()
    main(args)
