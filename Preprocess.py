import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import os

#warning for importing excel fil
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


# Configuration constants
INPUT_EXCEL = 'maybe_final/data/Arbejdsordre med afskrevet reservedele.xlsx'
#INPUT_CSV = 'data/trainingdatabentaxnew.csv'
MIN_SAMPLES_PER_PART = 5
PATTERN_BW = r"\bBW[34]\w*\b"

# Column names
WORK_ORDER_COL = 'Work Order'
INSTRUCTIONS_COL = 'Instructions'
PRODUCT_ID_COL = 'Product ID (Product) (Product)'
QUANTITY_COL = 'Quantity'
ASSET_PRODUCT_COL = 'Primær Asset Produkt'
WORK_ORDER_TYPE_COL = 'Work Order Type'


def load_data(path: str) -> pd.DataFrame:
    """
    Load Excel file into a DataFrame and log its shape.
    """
    df = pd.read_excel(path, engine="openpyxl")
    #df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
   
    # Mapping of actual column names in Excel to expected column names
    column_rename_map = {
        'Instructions (Work Order) (Work Order)': 'Instructions',
        'Primær Asset Produkt (Work Order) (Work Order)': 'Primær Asset Produkt',
        'Work Order Type (Work Order) (Work Order)': 'Work Order Type'
    }

    # Rename columns
    df = df.rename(columns=column_rename_map)

    return df


def group_workorders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group rows by Work Order and aggregate relevant columns.
    """
    agg_funcs = {
        INSTRUCTIONS_COL: lambda texts: ' '.join(
            str(t) for t in texts if pd.notna(t) and str(t).strip() != ''
        ),
        PRODUCT_ID_COL: lambda ids: [
            i for i in ids if pd.notna(i) and str(i).strip() != ''
        ],
        QUANTITY_COL: lambda qtys: [
            q for q in qtys if pd.notna(q)
        ],
        ASSET_PRODUCT_COL: lambda vals: ' '.join(
            pd.Series(vals).dropna().unique().astype(str)
        ),
        WORK_ORDER_TYPE_COL: 'first'
    }
    grouped = df.groupby(WORK_ORDER_COL).agg(agg_funcs).reset_index()
    print(f"Grouped into {len(grouped)} unique work orders.")
    return grouped


def print_filter_stats(before: int, after: int, description: str):
    """
    Print how many work orders were removed in a filter step.
    """
    removed = before - after
    print(f"Filter '{description}': removed {removed} work orders.")


def filter_workorders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sequential filters on the grouped DataFrame and log each step.
    """
    df_filtered = df.copy()
    print(f"Work orders before filtering: {len(df_filtered)}")

    # Filter 1: Only 'Nedbrud' work orders
    before = len(df_filtered)
    df_filtered = df_filtered[df_filtered[WORK_ORDER_TYPE_COL] == 'Nedbrud']
    print_filter_stats(before, len(df_filtered), "Work Order Type == 'Nedbrud'")

    # Filter 2: Asset product matches BW[3,4,c]
    before = len(df_filtered)
    df_filtered = df_filtered[
        df_filtered[ASSET_PRODUCT_COL].str.contains(PATTERN_BW, regex=True, na=False)
    ]
    print_filter_stats(before, len(df_filtered), f"Asset product matches {PATTERN_BW}")

    # Filter 3: Non-empty Instructions
    before = len(df_filtered)
    df_filtered = df_filtered[
        df_filtered[INSTRUCTIONS_COL].notna() &
        (df_filtered[INSTRUCTIONS_COL].str.strip() != '')
    ]
    print_filter_stats(before, len(df_filtered), "Non-empty Instructions")

    # Filter 4: Non-empty Product ID list
    before = len(df_filtered)
    df_filtered = df_filtered[
        df_filtered[PRODUCT_ID_COL].apply(lambda ids: len(ids) > 0)
    ]
    print_filter_stats(before, len(df_filtered), "Non-empty Product ID list")

    print(f"Work orders after filtering: {len(df_filtered)}")
    return df_filtered


def impute_quantity(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float], list[float]]:
    """
    Median imputation if quantity is 0 or below
    """

    # 0) Column added if missing
    if QUANTITY_COL not in df.columns:
        df[QUANTITY_COL] = np.nan
        print("Added missing 'Quantity' column with NaN values.")

    # 1) Flatten before changing
    pre_values = [q for sublist in df[QUANTITY_COL] for q in sublist]

    # 2) Median value
    observed_vals = [q for q in pre_values if pd.notna(q) and q != 0]
    if not observed_vals:
        print("Not enough valid data to calculate median.")
        return df, pre_values, pre_values

    median_val = int(round(np.median(observed_vals)))

    # 3) Impute NAN and 0 data
    def replace_missing(lst):
        return [median_val if (pd.isna(x) or x == 0) else x for x in lst]

    df[QUANTITY_COL] = df[QUANTITY_COL].apply(replace_missing)

    # 4) Flatten after change
    post_values = [q for sublist in df[QUANTITY_COL] for q in sublist]

    # 5) Logging
    replaced = sum(
        1 for before, after in zip(pre_values, post_values)
        if (pd.isna(before) or before == 0) and after == median_val
    )
    print(f"Imputed {replaced} 'Quantity' value(s) with median = {median_val:.2f}")

    return df, pre_values, post_values



def filter_min_samples(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """
    Remove work orders that contain any product with fewer than min_samples occurrences.
    """
    if min_samples <= 1:
        print(f"MIN_SAMPLES_PER_PART = {min_samples}, no filtering applied.")
        return df

    # Count product frequencies
    all_products = pd.Series([
        p for sublist in df[PRODUCT_ID_COL] for p in sublist
    ])
    counts = all_products.value_counts()
    rare = set(counts[counts < min_samples].index)

    before = len(df)
    df_filtered = df[~df[PRODUCT_ID_COL].apply(
        lambda ids: any(p in rare for p in ids)
    )]
    print_filter_stats(before, len(df_filtered), f"MIN_SAMPLES_PER_PART = {min_samples}")
    return df_filtered


def plot_quantity_distributions(pre_vals: list, post_vals: list):
    """
    Plot histograms of Quantity before and after imputation.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(pre_vals, bins=20)
    plt.title('Quantity Before Imputation')
    plt.xlabel('Quantity')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(post_vals, bins=20)
    plt.title('Quantity After Imputation')
    plt.xlabel('Quantity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_product_frequency(df: pd.DataFrame, top_n: int = 15):
    """
    Plot the top N most frequent Product IDs in the dataset.
    """
    all_products = pd.Series([
        p for sublist in df[PRODUCT_ID_COL] for p in sublist
    ])
    freq = all_products.value_counts().head(top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(freq.index.astype(str), freq.values)
    plt.xticks(rotation=90)
    plt.title(f'Top {top_n} Product ID Frequencies')
    plt.xlabel('Product ID')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def main():
    os.makedirs('Dataset', exist_ok=True)
    # Workflow steps
    #df = load_data(INPUT_CSV)
    df = load_data(INPUT_EXCEL)
    grouped = group_workorders(df)
    filtered = filter_workorders(grouped)
    imputed_df, pre_vals, post_vals = impute_quantity(filtered)
    #plot_quantity_distributions(pre_vals, post_vals)
    final_df = filter_min_samples(imputed_df, MIN_SAMPLES_PER_PART)
    fulldataset_df = final_df.copy() # copy to export full dataset
    #plot_product_frequency(final_df)

    # Split into train and test set (e.g., 80% train, 20% test)
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42, shuffle=True)
    
    # Save train and test files
    train_df.to_csv('Dataset/train_dataset.csv', index=False)
    test_df.to_csv('Dataset/test_dataset.csv', index=False)
    # Export full dataset
    fulldataset_df.to_csv('Dataset/full_dataset.csv', index=False)
    
    print(f"Train dataset saved to Dataset/train_dataset.csv, shape: {train_df.shape}")
    print(f"Test dataset saved to Dataset/test_dataset.csv, shape: {test_df.shape}")
    print(f"Full dataset saved to Dataset/full_dataset.csv, shape: {fulldataset_df.shape}")


if __name__ == '__main__':
    main()



### Jeg har lavet en ny version af preprocess.py, som er mere generel og kan bruges til at lave et dataset til en hvilken som helst type data. JEg har også sat visualiseringer på kommentar så den ikke bliver kørt hver gang.
