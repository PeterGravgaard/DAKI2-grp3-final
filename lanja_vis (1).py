import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

# === DATA ===
df = pd.read_csv("Dataset/full_dataset.csv")
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
df['product_id'] = df['product_id_(product)_(product)'].apply(ast.literal_eval)
df['quantity'] = df['quantity'].apply(ast.literal_eval)

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

# Simuler recall og precision
np.random.seed(42)
products = df_expanded['product_id'].unique()
recall = {pid: np.clip(np.random.normal(0.5, 0.15), 0, 1) for pid in products}
precision = {pid: np.clip(np.random.normal(0.3, 0.1), 0, 1) for pid in products}
df_expanded['recall'] = df_expanded['product_id'].map(recall)
df_expanded['precision'] = df_expanded['product_id'].map(precision)

# === VISUALISERING 1 ===
# Scatterplot: antal forekomster vs recall
product_freq = df_expanded['product_id'].value_counts().reset_index()
product_freq.columns = ['product_id', 'count']
product_freq['recall'] = product_freq['product_id'].map(recall)
product_freq['precision'] = product_freq['product_id'].map(precision)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='recall', color='blue')
plt.title('Recall vs. Product Frequency')
plt.xlabel('Antal forekomster af produkt')
plt.ylabel('Recall')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='precision', color='orange')
plt.title('Precision vs. Product Frequency')
plt.xlabel('Antal forekomster af produkt')
plt.ylabel('Precision')
plt.grid(True)
plt.tight_layout()
plt.show()


# === VISUALISERING 2 ===
# Instruction length pr. produkt_id vs recall OG precision – separat

length_perf = df_expanded.groupby('product_id').agg({
    'instruction_length': 'mean',
    'recall': 'mean',
    'precision': 'mean'
}).reset_index()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=length_perf, x='instruction_length', y='recall', color='blue')
plt.title('Instruction Length vs Recall per Product')
plt.xlabel('Gennemsnitlig længde af instruktion')
plt.ylabel('Recall')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=length_perf, x='instruction_length', y='precision', color='orange')
plt.title('Instruction Length vs Precision per Product')
plt.xlabel('Gennemsnitlig længde af instruktion')
plt.ylabel('Precision')
plt.grid(True)
plt.tight_layout()
plt.show()


# === VISUALISERING 3 ===
# Top 20 og bund 20 asset products baseret på recall
asset_perf = df_expanded.groupby('asset_product')['recall'].mean().reset_index()

top_20 = asset_perf.sort_values(by='recall', ascending=False).head(20)
bottom_20 = asset_perf.sort_values(by='recall', ascending=True).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(data=top_20, x='recall', y='asset_product', palette='Blues_d')
plt.title('Top 20 Asset Products by Average Recall')
plt.xlabel('Gennemsnitlig Recall')
plt.ylabel('Asset Product')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.barplot(data=bottom_20, x='recall', y='asset_product', palette='Reds_r')
plt.title('Bottom 20 Asset Products by Average Recall')
plt.xlabel('Gennemsnitlig Recall')
plt.ylabel('Asset Product')
plt.tight_layout()
plt.show()
