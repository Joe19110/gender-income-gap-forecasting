import pandas as pd
import os

# Automatically detect folder where this script is placed
base_folder = os.path.dirname(os.path.abspath(__file__))

# Dataset folder inside FODS
data_folder = os.path.join(base_folder, "dataset")

# Output file in FODS (not inside dataset)
output_file = os.path.join(base_folder, "cleaned_dataset.csv")

cleaned_dfs = []

# Loop through years and file suffix
for year in range(2014, 2024):
    for suffix in ["pusa", "pusb"]:
        
        file_path = os.path.join(data_folder, f"psam_{suffix}_{year}.csv")
        
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        print(f"Processing {file_path}...")
        df = pd.read_csv(file_path, low_memory=False)

        # Standardize columns
        col_mapping = {
            "ST": "ST",
            "STATEFIP": "ST",
            "STUSAB": "ST",
            "AGEP": "AGEP",
            "COW": "COW",
            "CLASSWKR": "COW",
            "SCHL": "SCHL",
            "SEX": "SEX",
            "WKHP": "WKHP",
            "PINCP": "PINCP",
            "OCCP": "OCCP"
        }

        # Keep only columns that exist
        existing_cols = [k for k in col_mapping if k in df.columns]
        df = df[existing_cols]

        # Rename to a consistent schema
        df.rename(columns={k: col_mapping[k] for k in existing_cols}, inplace=True)

        # Convert numeric columns
        numeric_cols = ["AGEP", "WKHP", "PINCP"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows missing essential values
        essential_cols = ["SEX", "PINCP", "AGEP", "ST", "WKHP"]
        essential_cols = [c for c in essential_cols if c in df.columns]
        df = df.dropna(subset=essential_cols)

        if "PINCP" in df.columns:
            df = df[df["PINCP"] > 0]

        # Fill missing categorical values
        if "SCHL" in df.columns:
            df["SCHL"] = df["SCHL"].fillna(df["SCHL"].mode()[0])

        if "COW" in df.columns:
            df["COW"] = df["COW"].fillna("Unemployed")

        if "OCCP" in df.columns:
            df["OCCP"] = df["OCCP"].fillna("Unknown")

        # Add readable labels
        if "SEX" in df.columns:
            df["Sex_Label"] = df["SEX"].map({1: "Male", 2: "Female"})

        if "COW" in df.columns:
            cow_mapping = {
                1:'Private for-profit',2:'Private non-profit',3:'Local government',4:'State government',
                5:'Federal government',6:'Self-employed not incorporated',7:'Self-employed incorporated',
                8:'Unpaid family worker'
            }
            df["COW"] = df["COW"].map(cow_mapping).fillna("Other/Unknown")

        if "OCCP" in df.columns:
            occupation_mapping = {
                10:'Manager/Business', 20:'Computer/Math', 30:'Engineering', 40:'Science',
                50:'Social Service', 60:'Legal', 70:'Education', 80:'Arts/Media',
                90:'Healthcare Practitioners', 100:'Healthcare Support', 110:'Protective Service',
                120:'Food Service', 130:'Maintenance', 140:'Personal Care', 150:'Sales',
                160:'Office/Admin', 170:'Farming/Fishing', 180:'Construction', 
                190:'Installation/Repair', 200:'Production', 210:'Transportation'
            }
            df["OCCP"] = df["OCCP"].map(occupation_mapping).fillna("Other/Unknown")

        # Rename final columns
        rename_cols = {
            "ST": "State_Code",
            "AGEP": "Age",
            "COW": "Class_of_Worker",
            "SCHL": "Education_Level",
            "WKHP": "Hours_Worked_Per_Week",
            "PINCP": "Personal_Income",
            "OCCP": "Occupation_Label"
        }

        df.rename(columns={c: rename_cols.get(c, c) for c in df.columns}, inplace=True)

        # Add year
        df["Year"] = year

        # Drop numeric SEX column, keep label
        if "SEX" in df.columns:
            df = df.drop(columns=["SEX"])

        cleaned_dfs.append(df)

# Combine everything
combined_df = pd.concat(cleaned_dfs, ignore_index=True)

# Save cleaned version
combined_df.to_csv(output_file, index=False)

print(f"\n✅ Combined cleaned dataset saved to: {output_file}")
print(f"📊 Total rows after cleaning: {len(combined_df)}")
