import json
import os
import sys


def check_json_names(directories):
    errors = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if 'exp_name' in data:
                                expected_name = data['exp_name']
                                actual_name = os.path.splitext(file)[0]
                                if expected_name != actual_name:
                                    errors.append(f"Error in {file_path}: exp_name '{expected_name}' does not match filename '{actual_name}'")
                    except json.JSONDecodeError as e:
                        errors.append(f"Error parsing {file_path}: {str(e)}")
                    except Exception as e:
                        errors.append(f"Error processing {file_path}: {str(e)}")
    
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    else:
        print("All JSON files validated successfully!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_json_names.py <directory1> [directory2] [directory3] ...")
        sys.exit(1)
    
    directories = sys.argv[1:]
    check_json_names(directories)
