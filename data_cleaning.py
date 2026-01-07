# data_cleaning.py
import pandas as pd
import numpy as np

def clean_data(file_path):
    """
    تنظيف البيانات من ملف Excel.
    """
    # تحميل البيانات
    df = pd.read_excel(file_path)
    
    # تحويل التاريخ
    df['activity_date'] = pd.to_datetime(df['activity_date'], errors='coerce')
    
    # معالجة opportunity_id: تحويل "no_opp" إلى NaN
    df['opportunity_id'] = df['opportunity_id'].replace('no_opp', np.nan)
    
    # إنشاء عمود النتيجة: 1 = فاز، 0 = خسر أو غير معروف
    df['outcome'] = df['opportunity_stage'].apply(lambda x: 1 if x == 'Won' else 0)
    
    # تنظيف الأعمدة الأساسية
    df = df.dropna(subset=['account_id', 'activity_date', 'types'])
    df = df[df['types'].notna()]
    
    return df

if __name__ == "__main__":
    df_clean = clean_data("data_all1.xlsx")
    df_clean.to_csv("cleaned_data.csv", index=False)
    print("✅ تم تنظيف البيانات وحفظها في 'cleaned_data.csv'")