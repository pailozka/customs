import pandas as pd
import numpy as np
import os

def clean_hs_code(code):
    if pd.isna(code):
        return None
    
    code_str = str(code).split('.')[0].strip()
    
    if not code_str:
        return None
        
    code_str = str(int(code_str)) if code_str.isdigit() else code_str
    return code_str.zfill(10)

def prepare_data():
    file_path = '/home/kara/customs/data_files/МОЯ МАРКИРОВКА ИЗГОТОВИТЕЛИ — копия.xlsx'
    print(f"Reading {file_path}...")
    
    # 1. Parse codes for embedding
    print("Parsing '!!!ИТС ВЕРНАЯ' sheet...")
    # header=2 treats the 3rd row as the header (0-indexed)
    df_codes = pd.read_excel(file_path, sheet_name='!!!ИТС ВЕРНАЯ', header=2)
    
    # We strip column names to be safe
    df_codes.columns = df_codes.columns.str.strip()
    
    if 'код' not in df_codes.columns and 'Код' in df_codes.columns:
        df_codes.rename(columns={'Код': 'код'}, inplace=True)
        
    df_codes = df_codes[['код', 'Наименование']].copy()
    df_codes.dropna(subset=['код', 'Наименование'], inplace=True)
    
    df_codes['code_clean'] = df_codes['код'].apply(clean_hs_code)
    df_codes.dropna(subset=['code_clean'], inplace=True)
    
    df_codes['description_clean'] = df_codes['Наименование'].astype(str).str.strip()
    
    # Deduplicate codes
    df_codes = df_codes.drop_duplicates(subset=['code_clean'])
    
    output_codes = df_codes[['code_clean', 'description_clean']].rename(
        columns={'code_clean': 'code', 'description_clean': 'description'}
    )
    
    output_codes.to_csv('codes_for_embedding.csv', index=False)
    print(f"Saved {len(output_codes)} unique codes to codes_for_embedding.csv")
    
    # 2. Parse test products
    print("Parsing 'Лист1' sheet...")
    df_test = pd.read_excel(file_path, sheet_name='Лист1', header=0)
    
    df_test.columns = df_test.columns.str.strip()
    
    df_test = df_test[['КОД ТНВЭД', 'наименование бирки', 'изготовитель']].copy()
    df_test.dropna(subset=['КОД ТНВЭД', 'наименование бирки'], inplace=True)
    
    df_test['ground_truth_code'] = df_test['КОД ТНВЭД'].apply(clean_hs_code)
    df_test.dropna(subset=['ground_truth_code'], inplace=True)
    
    df_test['product_name'] = df_test['наименование бирки'].astype(str).str.strip()
    df_test['manufacturer'] = df_test['изготовитель'].fillna('').astype(str).str.strip()
    
    output_test = df_test[['ground_truth_code', 'product_name', 'manufacturer']]
    
    if len(output_test) > 100:
        output_test = output_test.head(100)
        
    output_test.to_csv('products_test_set.csv', index=False)
    print(f"Saved {len(output_test)} test products to products_test_set.csv")

if __name__ == "__main__":
    prepare_data()
