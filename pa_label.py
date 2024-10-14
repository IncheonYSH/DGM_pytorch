import os
import argparse

def get_file_name(file_path):
    """
    Extract the file name from a given file path.

    Args:
        file_path (str): The full path of the file.

    Returns:
        str: The file name extracted from the path.
    """
    return os.path.basename(file_path)

def read_file(file_path):
    """
    Read a file and return the lines as a list of strings.

    Args:
        file_path (str): Path to the file to be read.

    Returns:
        list: List of strings, each representing a line in the file.
    """
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def write_to_file(file_path, data):
    """
    Write a list of strings to a file, one per line.

    Args:
        file_path (str): Path to the file to be written.
        data (list): List of strings to write to the file.
    """
    with open(file_path, 'w') as file:
        for line in data:
            file.write(f"{line}\n")

def process_files(pa_file, mimic_file, output_file):
    """
    Process the two input files, matching file names from the first file
    with paths from the second file and writing the matched paths to an output file.

    Args:
        pa_file (str): Path to the first input file.
        mimic_file (str): Path to the second input file.
        output_file (str): Path to the output file.
    """
    pa_list = read_file(pa_file)
    mimic_list = read_file(mimic_file)

    # Create a dictionary with file names from mimic_list as keys and full paths as values
    mimic_dict = {get_file_name(mimic_path): mimic_path for mimic_path in mimic_list}

    matched_paths = []
    
    for pa_path in pa_list:
        pa_file_name = get_file_name(pa_path)
        
        if pa_file_name in mimic_dict:
            # If the file name is found in mimic_file, append the full path from mimic_file to the output
            matched_paths.append(mimic_dict[pa_file_name])
        else:
            # If a match is not found, print a warning and stop the process
            print(f"Warning: File '{pa_file_name}' from the first file is not found in the second file.")
            return

    # Write the matched paths to the output file
    write_to_file(output_file, matched_paths)
    print(f"Matched paths have been written to {output_file}")

# python pa_label.py --pa_file <pa_test.txt 경로> --mimic_file <test.txt 경로> --output_file <output.txt 경로>
# python pa_label.py --pa_file ./data/pa_train.txt --mimic_file ./data/train.txt --output_file ./data/pa_label_train.txt
# python pa_label.py --pa_file ./data/pa_validation.txt --mimic_file ./data/validation.txt --output_file ./data/pa_label_validation.txt
# python pa_label.py --pa_file ./data/pa_test.txt --mimic_file ./data/test.txt --output_file ./data/pa_label_test.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two file lists and write matched file paths to an output file.")
    parser.add_argument('--pa_file', type=str, required=True, help="Path to the first file (e.g., pa_test.txt).")
    parser.add_argument('--mimic_file', type=str, required=True, help="Path to the second file (e.g., test.txt).")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output file where matched paths will be saved.")
    
    args = parser.parse_args()

    process_files(args.pa_file, args.mimic_file, args.output_file)
