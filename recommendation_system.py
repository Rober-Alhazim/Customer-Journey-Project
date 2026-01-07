# recommendation_system.py
import pandas as pd
import numpy as np
import joblib

def load_components():
    dt = joblib.load("decision_tree_model.pkl")
    le_country = joblib.load("label_encoder_country.pkl")
    le_solution = joblib.load("label_encoder_solution.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return dt, le_country, le_solution, feature_cols

def recommend_next_actions(country, solution, current_counts=None):
    """
    توصية بأفضل 4 إجراءات باستخدام نموذج Decision Tree.
    """
    dt, le_country, le_solution, feature_cols = load_components()
    
    if current_counts is None:
        current_counts = {}
    
    # بناء المتجه الحالي
    base = {}
    for col in feature_cols:
        if col.endswith('_count'):
            act = col.replace('_count', '')
            base[col] = current_counts.get(act, 0)
        elif col == 'Country_encoded':
            base[col] = le_country.transform([country])[0]
        elif col == 'solution_encoded':
            base[col] = le_solution.transform([solution])[0]
    
    recommendations = []
    for col in feature_cols:
        if col.endswith('_count'):
            test_vec = base.copy()
            test_vec[col] += 1
            prob = dt.predict_proba(pd.DataFrame([test_vec]))[0][1]
            act_name = col.replace('_count', '')
            recommendations.append((act_name, prob))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:4]

def get_top4_from_analysis(country, solution):
    """
    جلب أفضل 4 إجراءات من التحليل الإحصائي (الملفات المحفوظة).
    """
    top_country = pd.read_csv("top_by_country.csv")
    top_solution = pd.read_csv("top_by_solution.csv")
    top_both = pd.read_csv("top_by_country_solution.csv")
    
    print(f"\n=== التوصيات لـ {country} + {solution} ===")
    
    print("\n1. حسب البلد:")
    res1 = top_country[top_country['Country'] == country].nlargest(4, 'count')
    print(res1[['types', 'count']].to_string(index=False))
    
    print("\n2. حسب الحل:")
    res2 = top_solution[top_solution['solution'] == solution].nlargest(4, 'count')
    print(res2[['types', 'count']].to_string(index=False))
    
    print("\n3. حسب البلد والحل معًا:")
    res3 = top_both[
        (top_both['Country'] == country) &
        (top_both['solution'] == solution)
    ].nlargest(4, 'count')
    if not res3.empty:
        print(res3[['types', 'count']].to_string(index=False))
    else:
        print(" - لا توجد سجلات كافية لهذا المزيج.")

def main_recommendation(country, solution, new_action=None):
    """
    عرض جميع أنواع التوصيات.
    """
    # تحديث current_counts إذا تم إدخال إجراء جديد
    current_counts = {}
    if new_action:
        current_counts[new_action] = 1
    
    # 1. التوصيات الإحصائية
    get_top4_from_analysis(country, solution)
    
    # 2. التوصيات باستخدام Decision Tree
    print("\n4. باستخدام Decision Tree (احتمال الفوز):")
    recs = recommend_next_actions(country, solution, current_counts)
    for action, prob in recs:
        print(f"   - {action}: {prob:.2%}")

if __name__ == "__main__":
    # مثال على استخدام النظام
    country = "US"
    solution = "MRS"
    new_action = None  # أو "Email" مثلاً

    main_recommendation(country, solution, new_action)