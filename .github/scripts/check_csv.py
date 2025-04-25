#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd


def validate_csv(file_path):
    """Validate a CSV file by checking if it can be properly parsed."""
    print(f"Validating {file_path}...")
    try:
        # Try to read the CSV with pandas
        df = pd.read_csv(file_path)
        
        # Basic structural checks
        if df.empty:
            print(f"❌ Error: {file_path} is empty")
            return False
        
        print(f"✅ {file_path} passed validation")
        return True
        
    except pd.errors.ParserError as e:
        print(f"❌ Error: {file_path} has parsing errors: {str(e)}")
        return False
    except UnicodeDecodeError:
        print(f"❌ Error: {file_path} has encoding issues. Try using UTF-8 encoding.")
        return False
    except Exception as e:
        print(f"❌ Error validating {file_path}: {str(e)}")
        return False

def main():
    # Find all CSV files in the current directory (non-recursive)
    csv_files = list(Path('.').glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in the project root folder")
        return 0
    
    print(f"Found {len(csv_files)} CSV files to validate")
    
    failed = False
    validated_count = 0
    failed_count = 0
    
    for csv_file in csv_files:
        if validate_csv(csv_file):
            validated_count += 1
        else:
            failed = True
            failed_count += 1
    
    print("\n=== Validation Summary ===")
    print(f"Total CSV files: {len(csv_files)}")
    print(f"Passed: {validated_count}")
    print(f"Failed: {failed_count}")
    
    if failed:
        print("❌ Validation failed. Please fix the reported issues.")
        return 1
    else:
        print("✅ All CSV files passed validation!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
