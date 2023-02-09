import shutil
import glob
import os
import sys


def merge_csv_files(path, pattern="*.tsv", result_file_name="merged.tsv"):
    root_dir_path=os.path.dirname(sys.modules['__main__'].__file__)
    allFiles = glob.glob(f"{root_dir_path}/{path}/{pattern}")
    allFiles.sort()
    print(f"Merging files: {allFiles}")
    with open(f"{path}/{result_file_name}", "wb") as outfile:
        for i, fname in enumerate(allFiles):
            with open(fname, "rb") as infile:
                if i != 0:
                    infile.readline()  # Throw away header on all but first file
                shutil.copyfileobj(infile, outfile)
