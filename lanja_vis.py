import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import os.path

# === DATA ===
df = pd.read_csv("Dataset/full_dataset.csv")
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
df['product_id'] = df['product_id_(product)_(product)'].apply(ast.literal_eval)
df['quantity'] = df['quantity'].apply(ast.literal_eval)

# Tjek om produkt-performance data findes, hvis ikke, kør test_testset maj 20.py først
product_performance_path = "Dataset/product_performance.csv"
if not os.path.exists(product_performance_path):
    print("FEJL: Product performance data findes ikke.")
    print("Kør 'python test_testset\\ maj\\ 20.py' først for at generere data.")
    exit(1)

# Indlæs rigtig recall og precision data
product_perf = pd.read_csv(product_performance_path)
print(f"Indlæst performance data for {len(product_perf)} produkter")

# Udvid datasættet: én række pr. produkt
rows = []
for _, row in df.iterrows():
    for pid, qty in zip(row['product_id'], row['quantity']):
        rows.append({
            'product_id': pid,
            'instructions': row['instructions'],
            'asset_product': row['primær_asset_produkt'],
            'instruction_length': len(str(row['instructions']).split())
        })
df_expanded = pd.DataFrame(rows)

# Slå recall og precision op fra rigtige data i stedet for simuleret data
product_lookup = product_perf.set_index('product_id').to_dict()
df_expanded['recall'] = df_expanded['product_id'].map(product_lookup['recall'])
df_expanded['precision'] = df_expanded['product_id'].map(product_lookup['precision'])

# Håndter produkter der ikke har performance data (kan være i træningssæt men ikke i test)
missing_mask = df_expanded['recall'].isna()
if missing_mask.any():
    print(f"Bemærk: {missing_mask.sum()} forekomster mangler performance data")
    df_expanded = df_expanded.dropna(subset=['recall', 'precision'])

# === VISUALISERING 1 ===
# Scatterplot: antal forekomster vs recall
product_freq = df_expanded.groupby('product_id').size().reset_index(name='dataset_count')
# Merge with product_perf, ensuring we don't lose the 'count' column from product_perf
product_freq = product_freq.merge(product_perf, on='product_id', how='inner')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='recall', color='blue', alpha=0.7)
plt.title('Recall vs. Product Frequency (Rigtige Data)')
plt.xlabel('Antal forekomster af produkt (ground truth)')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Tilføj trend linje
plt.figure(figsize=(10, 6))
sns.regplot(data=product_freq, x='count', y='recall', 
           scatter_kws={'alpha': 0.5, 'color': 'blue'}, 
           line_kws={'color': 'red'})
plt.title('Recall vs. Product Frequency med Trend Linje')
plt.xlabel('Antal forekomster af produkt (ground truth)')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='precision', color='orange', alpha=0.7)
plt.title('Precision vs. Product Frequency (Rigtige Data)')
plt.xlabel('Antal forekomster af produkt (ground truth)')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Tilføj trend linje for precision
plt.figure(figsize=(10, 6))
sns.regplot(data=product_freq, x='count', y='precision', 
           scatter_kws={'alpha': 0.5, 'color': 'orange'}, 
           line_kws={'color': 'red'})
plt.title('Precision vs. Product Frequency med Trend Linje')
plt.xlabel('Antal forekomster af produkt (ground truth)')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# === VISUALISERING 2 ===
# Instruction length pr. produkt_id vs recall OG precision
length_perf = df_expanded.groupby('product_id').agg({
    'instruction_length': 'mean',
    'recall': 'mean',
    'precision': 'mean'
}).reset_index()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=length_perf, x='instruction_length', y='recall', color='blue', alpha=0.7)
plt.title('Instruction Length vs Recall per Product (Rigtige Data)')
plt.xlabel('Gennemsnitlig længde af instruktion')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Tilføj trend linje
plt.figure(figsize=(10, 6))
sns.regplot(data=length_perf, x='instruction_length', y='recall', 
           scatter_kws={'alpha': 0.5, 'color': 'blue'}, 
           line_kws={'color': 'red'})
plt.title('Instruction Length vs Recall med Trend Linje')
plt.xlabel('Gennemsnitlig længde af instruktion')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=length_perf, x='instruction_length', y='precision', color='orange', alpha=0.7)
plt.title('Instruction Length vs Precision per Product (Rigtige Data)')
plt.xlabel('Gennemsnitlig længde af instruktion')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Tilføj trend linje
plt.figure(figsize=(10, 6))
sns.regplot(data=length_perf, x='instruction_length', y='precision', 
           scatter_kws={'alpha': 0.5, 'color': 'orange'}, 
           line_kws={'color': 'red'})
plt.title('Instruction Length vs Precision med Trend Linje')
plt.xlabel('Gennemsnitlig længde af instruktion')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# === VISUALISERING 3 ===
# Top 20 og bund 20 asset products baseret på recall
asset_perf = df_expanded.groupby('asset_product')['recall'].mean().reset_index()
asset_counts = df_expanded.groupby('asset_product').size().reset_index(name='count')
asset_perf = asset_perf.merge(asset_counts, on='asset_product')

# Filtrer evt. for assets med tilstrækkeligt antal forekomster
min_count = 5  # Minimum antal forekomster for at inkludere et asset
asset_perf_filtered = asset_perf[asset_perf['count'] >= min_count]

top_20 = asset_perf_filtered.sort_values(by='recall', ascending=False).head(20)
bottom_20 = asset_perf_filtered.sort_values(by='recall', ascending=True).head(20)

plt.figure(figsize=(12, 10))
bars = sns.barplot(data=top_20, x='recall', y='asset_product', palette='Blues_d')
plt.title('Top 20 Asset Products by Average Recall (Rigtige Data)')
plt.xlabel('Gennemsnitlig Recall')
plt.ylabel('Asset Product')
# Tilføj antal forekomster på hver bar
for i, row in enumerate(top_20.itertuples()):
    plt.text(0.01, i, f"n={row.count}", va='center')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
bars = sns.barplot(data=bottom_20, x='recall', y='asset_product', palette='Reds_r')
plt.title('Bottom 20 Asset Products by Average Recall (Rigtige Data)')
plt.xlabel('Gennemsnitlig Recall')
plt.ylabel('Asset Product')
# Tilføj antal forekomster på hver bar
for i, row in enumerate(bottom_20.itertuples()):
    plt.text(0.01, i, f"n={row.count}", va='center')
plt.tight_layout()
plt.show()


# === EKSTRA VISUALISERING ===
# F1-score beregning og visualisering
product_freq['f1'] = 2 * (product_freq['precision'] * product_freq['recall']) / (product_freq['precision'] + product_freq['recall'])

plt.figure(figsize=(10, 6))
sns.regplot(data=product_freq, x='count', y='f1', 
           scatter_kws={'alpha': 0.6, 'color': 'green'}, 
           line_kws={'color': 'red'})
plt.title('F1-Score vs. Product Frequency med Trend Linje')
plt.xlabel('Antal forekomster af produkt (ground truth)')
plt.ylabel('F1-Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Gem visualiseringer til filer
# plt.figure(figsize=(10, 6))
# sns.regplot(data=product_freq, x='count', y='recall', 
#            scatter_kws={'alpha': 0.5, 'color': 'blue'}, 
#            line_kws={'color': 'red'})
# plt.title('Recall vs. Product Frequency med Trend Linje')
# plt.xlabel('Antal forekomster af produkt')
# plt.ylabel('Recall')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('Visual/recall_vs_frequency.png', dpi=300, bbox_inches='tight')
# plt.close()
