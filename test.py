from pathlib import Path
import pandas as pd

df = pd.read_pickle('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/loud_and_whisper_and_lip_20250713_064722.pkl')

print("=" * 60)
print("DATA ANALYSIS")
print("=" * 60)

print(f"\nTotal number of rows: {len(df)}")

print("\nDataFrame columns:")
print(df.columns.tolist())

# Check if there's an 'instruction' or similar column
if 'instruction' in df.columns:
    print("\nInstruction type counts:")
    print(df['instruction'].value_counts())
    print(f"\nTotal unique instruction types: {df['instruction'].nunique()}")
elif 'instruction_type' in df.columns:
    print("\nInstruction type counts:")
    print(df['instruction_type'].value_counts())
    print(f"\nTotal unique instruction types: {df['instruction_type'].nunique()}")
else:
    print("\nSearching for instruction-related columns...")
    instruction_cols = [col for col in df.columns if 'instruction' in col.lower() or 'type' in col.lower()]
    if instruction_cols:
        print(f"Found columns: {instruction_cols}")
        for col in instruction_cols:
            print(f"\n{col} value counts:")
            print(df[col].value_counts())
    else:
        print("No instruction-related columns found.")
        print("\nFirst few rows:")
        print(df.head())

print("\n" + "=" * 60)