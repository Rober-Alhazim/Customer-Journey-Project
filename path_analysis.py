# path_analysis.py
import pandas as pd

def analyze_top_paths(df):
    """
    تحليل أفضل 4 إجراءات (actions) حسب:
    - البلد
    - الحل
    - البلد + الحل
    """
    # فقط الفرص الناجحة (أو جميعها إذا أردت)
    df_won = df[df['opportunity_stage'] == 'Won'].copy()
    if df_won.empty:
        df_won = df  # استخدام كل البيانات إذا لم توجد "Won"

    def get_top_4(groupby_cols):
        return (
            df_won.groupby(groupby_cols)['types']
            .value_counts()
            .groupby(groupby_cols)
            .head(4)
            .reset_index(name='count')
        )

    top_by_country = get_top_4(['Country'])
    top_by_solution = get_top_4(['solution'])
    top_by_country_solution = get_top_4(['Country', 'solution'])

    return top_by_country, top_by_solution, top_by_country_solution

if __name__ == "__main__":
    from data_cleaning import clean_data
    df = clean_data("data_all1.xlsx")
    top_country, top_solution, top_both = analyze_top_paths(df)
    
    top_country.to_csv("top_by_country.csv", index=False)
    top_solution.to_csv("top_by_solution.csv", index=False)
    top_both.to_csv("top_by_country_solution.csv", index=False)
    
    print("✅ تم تحليل المسارات وحفظ النتائج.")