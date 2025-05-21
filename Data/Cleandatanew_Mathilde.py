import pandas as pd 
import re
from sklearn.model_selection import train_test_split

# 1. Læs Excel-fil
file_path = r'Data\~$Arbejdsordre med afskrevet reservedele.xlsx'
df = pd.read_excel(file_path, engine="openpyxl")

# 2. Udskriv kolonner før ændringer
print(" Kolonner før ændringer:")
for idx, col in enumerate(df.columns):
    print(f"{idx}: {col}")

# 3. Fjern kolonner der indeholder 'Do Not Modify'
df = df.loc[:, ~df.columns.str.contains('Do Not Modify')]

# 4. Fjern "(Work Order)" og lignende fra kolonnenavne
df.columns = df.columns.str.replace(r' \(Work Order\)', '', regex=True)
df.columns = df.columns.str.replace(r' \(Work Order Incident\)', '', regex=True)

# 5. Udskriv kolonner efter renaming
print("\n Kolonner efter renaming:")
for idx, col in enumerate(df.columns):
    print(f"{idx}: {col}")

# 6. Vælg relevante kolonner
columns_to_keep = [
    'Work Order',
    'Postal Code',
    'First Arrived On',
    'Completed On',
    'Primary Incident Customer Asset',
    'Customer Asset',
    'Line Status',
    'Incident Type',
    'Primær Asset Kategori',
    'Primær Asset Produkt',
    'Work Order Type',
    'Primary Incident Type',
    'Instructions',
    'Intern note',
    'Koptæller',
    'Product ID (Product) (Product)',
    'Supplier Item number (Product) (Product)',
    'Name',
    'Quantity'
]

# 7. Tjek for manglende kolonner
missing_cols = [col for col in columns_to_keep if col not in df.columns]
if missing_cols:
    print(f" Følgende kolonner blev ikke fundet og fjernes fra udvalget: {missing_cols}")
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]

# 8. Vælg kun de ønskede kolonner
df = df[columns_to_keep]

# 9. Fjern rækker med manglende 'Instructions' eller 'Product ID'
df = df.dropna(subset=['Instructions', 'Product ID (Product) (Product)'])

# 10. Filtrér kun 'Work Order Type' == 'Nedbrud'
df = df[df['Work Order Type'] == 'Nedbrud']

# 11. Filtrér rækker med BW3, BW4 og BWc modeller
pattern = r'\bBW[34c]\w*\b'
df_filtered_bw_models = df[df['Primær Asset Produkt'].str.contains(pattern, case=False, na=False)]

# 12. Split i træning og test (80/20)
train_df, test_df = train_test_split(df_filtered_bw_models, test_size=0.2, random_state=42)

# 13. Sortér begge datasæt efter 'Work Order'
train_df = train_df.sort_values(by='Work Order')
test_df = test_df.sort_values(by='Work Order')

# 14. Tjek unikke 'Work Order Type' efter split
print("\n Unikke 'Work Order Type' i træningssættet:")
print(train_df['Work Order Type'].unique())

print("\n Unikke 'Work Order Type' i testsættet:")
print(test_df['Work Order Type'].unique())

# 15. Gem som CSV
train_df.to_csv('trainingdatabentaxnew.csv', index=False)
test_df.to_csv('testdatabentaxnew.csv', index=False)

# 16. Udskriv resultater
print(f"\n Færdig!\nTraining set size: {len(train_df)}\nTest set size: {len(test_df)}")
