# decision_tree_model.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def build_decision_tree(df):
    """
    بناء نموذج Decision Tree لتحديد أهم الإجراءات المؤثرة على الفوز.
    """
    # حصر البيانات بالفرص (التي لها opportunity_id)
    df_opps = df[df['opportunity_id'].notna()].copy()
    
    # تحديد أنواع الأنشطة الفريدة
    activity_types = df_opps['types'].unique()
    
    # إنشاء أعمدة عدد كل نوع من الأنشطة
    for act in activity_types:
        df_opps[act + '_count'] = (df_opps['types'] == act).astype(int)
    
    # تجميع حسب الفرصة
    features = df_opps.groupby('opportunity_id').agg(
        Country=('Country', 'first'),
        solution=('solution', 'first'),
        outcome=('outcome', 'max'),
        **{act + '_count': (act + '_count', 'sum') for act in activity_types}
    ).reset_index()
    
    # ترميز المتغيرات الفئوية
    le_country = LabelEncoder()
    le_solution = LabelEncoder()
    
    features['Country_encoded'] = le_country.fit_transform(features['Country'])
    features['solution_encoded'] = le_solution.fit_transform(features['solution'])
    
    # تحديد الأعمدة المستخدمة
    feature_cols = [col for col in features.columns if col.endswith('_count')] + ['Country_encoded', 'solution_encoded']
    X = features[feature_cols]
    y = features['outcome']
    
    # تدريب النموذج
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X, y)
    
    # حفظ النموذج والمشفرات
    joblib.dump(dt, "decision_tree_model.pkl")
    joblib.dump(le_country, "label_encoder_country.pkl")
    joblib.dump(le_solution, "label_encoder_solution.pkl")
    joblib.dump(feature_cols, "feature_columns.pkl")
    
    # طباعة أهم الميزات
    importances = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nأبرز الإجراءات المؤثرة على الفوز:")
    print(importances.head(10))
    
    return dt, le_country, le_solution, feature_cols

if __name__ == "__main__":
    from data_cleaning import clean_data
    df = clean_data("data_all1.xlsx")
    build_decision_tree(df)
    print("✅ تم تدريب نموذج Decision Tree وحفظه.")