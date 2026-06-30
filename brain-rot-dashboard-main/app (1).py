import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# الإعدادات العامة للصفحة
# =========================================================
st.set_page_config(
    page_title="Brain Rot Dashboard | تحليل تأثير السوشل ميديا",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# تنسيق CSS مخصص (هوية بصرية احترافية)
# =========================================================
st.markdown("""
<style>
.main { background-color: #0f1116; }
.metric-card {
    background: linear-gradient(135deg, #1c1f2b 0%, #262b3d 100%);
    padding: 18px 22px; border-radius: 14px;
    border: 1px solid #353b52;
}
h1, h2, h3 { color: #f5f5f7; }
.insight-box {
    background-color: #1a2332; border-left: 4px solid #5b8def;
    padding: 14px 18px; border-radius: 8px; margin: 8px 0;
    color: #d8dee9; font-size: 15px;
}
.warn-box {
    background-color: #2a1a1a; border-left: 4px solid #e05c5c;
    padding: 14px 18px; border-radius: 8px; margin: 8px 0;
    color: #f1d8d8; font-size: 15px;
}
[data-testid="stMetricValue"] { font-size: 26px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# تحميل البيانات + هندسة المتغيرات
# =========================================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    usage_bins = [0, 3.8, 6.3, 9]
    usage_labels = ['Light Student', 'Moderate Student', 'Heavy Student']
    df['Student_Category'] = pd.cut(df['Avg_Daily_Usage_Hours'], bins=usage_bins, labels=usage_labels)

    stress_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    df['Stress_Level_Numeric'] = df['Stress_Level'].map(stress_map)

    df['Brain_Rot_Score'] = (df['Daily_Unlocks'] / 30) + (df['Avg_Daily_Usage_Hours'] / 2)
    df['Brain_Rot_Score'] = df['Brain_Rot_Score'].clip(1, 10).round(1)

    return df

st.sidebar.title("🧠 Brain Rot Analytics")
st.sidebar.caption("لوحة تحليل أثر السوشل ميديا على الصحة النفسية للطلاب")

DEFAULT_DATA_PATH = "data.csv"

import os
data_source = None
if os.path.exists(DEFAULT_DATA_PATH):
    data_source = DEFAULT_DATA_PATH
    st.sidebar.success("✅ البيانات محمّلة تلقائيًا (data.csv)")
else:
    uploaded_file = st.sidebar.file_uploader("📂 رفع ملف البيانات (CSV)", type=["csv"])
    data_source = uploaded_file

if data_source is None:
    st.title("🧠 Student Social Media & Mental Health — Brain Rot Dashboard")
    st.info("⬅️ ابدأ برفع ملف **Student Social Media And Mental Health Impact.csv** من الشريط الجانبي لعرض الداشبورد.")
    st.stop()

df = load_data(data_source)

# =========================================================
# الفلاتر الجانبية — الفلتر الرئيسي: Stress_Level
# =========================================================
st.sidebar.markdown("### 🎛️ فلاتر التحكم")

stress_options = ['Low', 'Medium', 'High', 'Very High']
selected_stress = st.sidebar.multiselect(
    "مستوى التوتر (Stress Level)", stress_options, default=stress_options
)

selected_category = st.sidebar.multiselect(
    "فئة الاستخدام", df['Student_Category'].dropna().unique().tolist(),
    default=df['Student_Category'].dropna().unique().tolist()
)

selected_platforms = st.sidebar.multiselect(
    "المنصة الأكثر استخدامًا", sorted(df['Most_Used_Platform'].unique().tolist()),
    default=sorted(df['Most_Used_Platform'].unique().tolist())
)

age_range = st.sidebar.slider(
    "النطاق العمري", int(df['Age'].min()), int(df['Age'].max()),
    (int(df['Age'].min()), int(df['Age'].max()))
)

filtered = df[
    (df['Stress_Level'].isin(selected_stress)) &
    (df['Student_Category'].isin(selected_category)) &
    (df['Most_Used_Platform'].isin(selected_platforms)) &
    (df['Age'].between(age_range[0], age_range[1]))
]

st.sidebar.markdown("---")
st.sidebar.caption(f"📊 عدد السجلات المعروضة: **{len(filtered):,}** من أصل {len(df):,}")

if filtered.empty:
    st.warning("⚠️ لا توجد بيانات تطابق الفلاتر المختارة. عدّل الفلاتر من الشريط الجانبي.")
    st.stop()

# =========================================================
# العنوان + نظرة تحليلية تنفيذية
# =========================================================
st.title("🧠 Student Social Media & Mental Health — Brain Rot Dashboard")
st.caption("تحليل تفاعلي بيانات وذكاء اصطناعي لأثر استخدام السوشل ميديا على الصحة النفسية للطلاب")

# =========================================================
# مؤشرات الأداء الرئيسية KPIs
# =========================================================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("متوسط ساعات الاستخدام اليومي", f"{filtered['Avg_Daily_Usage_Hours'].mean():.1f} ساعة")
k2.metric("متوسط ساعات النوم", f"{filtered['Sleep_Hours_Per_Night'].mean():.1f} ساعة")
k3.metric("متوسط درجة الصحة النفسية", f"{filtered['Mental_Health_Score'].mean():.1f} / 10")
k4.metric("متوسط درجة تعفن الدماغ", f"{filtered['Brain_Rot_Score'].mean():.1f} / 10")
high_stress_pct = (filtered['Stress_Level'].isin(['High', 'Very High']).mean()) * 100
k5.metric("نسبة التوتر المرتفع", f"{high_stress_pct:.0f}%")

st.markdown("---")

# =========================================================
# Tabs رئيسية
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 نظرة عامة", "🔍 تحليل العلاقات", "🤖 نموذج التنبؤ بالصحة النفسية",
    "🌲 تصنيف مستوى التوتر", "📝 الملخص التنفيذي"
])

# ---------------------------------------------------------
# TAB 1: نظرة عامة
# ---------------------------------------------------------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            filtered, x="Avg_Daily_Usage_Hours", nbins=20,
            color_discrete_sequence=["#5b8def"],
            title="توزيع ساعات الاستخدام اليومي"
        )
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        stress_counts = filtered['Stress_Level'].value_counts().reindex(stress_options).fillna(0)
        fig = px.bar(
            x=stress_counts.index, y=stress_counts.values,
            color=stress_counts.index,
            color_discrete_sequence=px.colors.sequential.RdBu,
            title="توزيع مستويات التوتر", labels={"x": "مستوى التوتر", "y": "عدد الطلاب"}
        )
        fig.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        gender_counts = filtered['Gender'].value_counts()
        fig = px.pie(
            names=gender_counts.index, values=gender_counts.values,
            title="توزيع الجنس", hole=0.45,
            color_discrete_sequence=["#5b8def", "#e8825a"]
        )
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        platform_gender = filtered.groupby(['Most_Used_Platform', 'Gender']).size().reset_index(name='Count')
        fig = px.bar(
            platform_gender, x="Most_Used_Platform", y="Count", color="Gender",
            barmode="group", title="المنصات الأكثر استخدامًا حسب الجنس",
            color_discrete_sequence=["#5b8def", "#e8825a"]
        )
        fig.update_layout(template="plotly_dark", height=380, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: تحليل العلاقات
# ---------------------------------------------------------
with tab2:
    c1, c2 = st.columns([1.3, 1])

    with c1:
        numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Study_Hours',
                         'Physical_Activity_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score',
                         'Stress_Level_Numeric', 'Brain_Rot_Score']
        corr = filtered[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            title="مصفوفة الارتباط بين المتغيرات", aspect="auto"
        )
        fig.update_layout(template="plotly_dark", height=520)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            filtered, x="Stress_Level", y="Sleep_Hours_Per_Night",
            category_orders={"Stress_Level": stress_options},
            color="Stress_Level",
            color_discrete_sequence=px.colors.sequential.RdBu,
            title="ساعات النوم حسب مستوى التوتر"
        )
        fig.update_layout(template="plotly_dark", height=520, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(
            filtered, x="Avg_Daily_Usage_Hours", y="Mental_Health_Score",
            color="Stress_Level", category_orders={"Stress_Level": stress_options},
            color_discrete_sequence=px.colors.sequential.RdBu,
            title="ساعات الاستخدام مقابل الصحة النفسية", opacity=0.6
        )
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.box(
            filtered, x="Student_Category", y="Brain_Rot_Score",
            color="Student_Category",
            category_orders={"Student_Category": ["Light Student", "Moderate Student", "Heavy Student"]},
            color_discrete_sequence=px.colors.sequential.RdBu,
            title="درجة تعفن الدماغ حسب فئة الاستخدام"
        )
        fig.update_layout(template="plotly_dark", height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: نموذج التنبؤ (Linear Regression تفاعلي)
# ---------------------------------------------------------
with tab3:
    st.subheader("🤖 نموذج الانحدار الخطي — التنبؤ بدرجة الصحة النفسية")
    st.caption("Linear Regression: التنبؤ بدرجة رقمية مستمرة (Mental_Health_Score)")

    X = filtered[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']]
    y = filtered['Mental_Health_Score']

    if len(filtered) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("R² (دقة التفسير)", f"{r2:.3f}")
        m2.metric("RMSE (خطأ التنبؤ)", f"{rmse:.3f}")
        m3.metric("عدد بيانات التدريب", f"{len(X_train):,}")

        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = px.scatter(
                x=y_test, y=y_pred, opacity=0.6,
                labels={"x": "القيمة الفعلية", "y": "القيمة المتوقعة"},
                title="القيم الفعلية مقابل المتوقعة", color_discrete_sequence=["#5b8def"]
            )
            min_v, max_v = float(y_test.min()), float(y_test.max())
            fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                      mode="lines", line=dict(color="#e05c5c", dash="dash"),
                                      name="خط التنبؤ المثالي"))
            fig.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 🎯 جرّب التنبؤ بنفسك")
            usage_input = st.slider("ساعات الاستخدام اليومي", 0.0, 12.0, 5.0, 0.1)
            sleep_input = st.slider("ساعات النوم اليومي", 0.0, 12.0, 7.0, 0.1)
            pred = model.predict([[usage_input, sleep_input]])[0]
            pred = float(np.clip(pred, 0, 10))

            st.markdown(f"""
            <div class="metric-card" style="text-align:center; margin-top:10px;">
                <p style="color:#9aa3b8; margin-bottom:4px;">الصحة النفسية المتوقعة</p>
                <h1 style="color:#5b8def; margin:0;">{pred:.1f} / 10</h1>
            </div>
            """, unsafe_allow_html=True)

            if pred >= 7:
                st.markdown('<div class="insight-box">✅ مستوى صحة نفسية جيد — استمر بهذا التوازن.</div>', unsafe_allow_html=True)
            elif pred >= 5:
                st.markdown('<div class="insight-box">⚠️ مستوى متوسط — يُنصح بتقليل الاستخدام أو زيادة ساعات النوم.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">🚨 مستوى منخفض — يُنصح بمراجعة نمط الاستخدام والنوم بشكل عاجل.</div>', unsafe_allow_html=True)
    else:
        st.warning("عدد البيانات المفلترة قليل جدًا لتدريب نموذج موثوق. وسّع نطاق الفلاتر.")

# ---------------------------------------------------------
# TAB 4: تصنيف مستوى التوتر (Decision Tree vs Random Forest)
# ---------------------------------------------------------
with tab4:
    st.subheader("🌲 تصنيف مستوى التوتر — Decision Tree vs Random Forest")
    st.caption("التنبؤ بفئة التوتر (Low / Medium / High / Very High) بناءً على: ساعات الدراسة، ساعات الاستخدام، عدد فتحات الهاتف")

    X1 = filtered[['Study_Hours', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks']]
    y1 = filtered['Stress_Level_Numeric']

    if len(filtered) >= 20 and y1.nunique() >= 2:
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

        # نموذج 1: Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train1, y_train1)
        y_pred_dt = dt_model.predict(X_test1)

        # نموذج 2: Random Forest
        rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
        rf_model.fit(X_train1, y_train1)
        y_pred_rf = rf_model.predict(X_test1)

        def get_metrics(y_true, y_pred):
            return {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "F1-score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            }

        dt_metrics = get_metrics(y_test1, y_pred_dt)
        rf_metrics = get_metrics(y_test1, y_pred_rf)

        st.markdown("#### 📊 مقارنة أداء النموذجين")
        comp_df = pd.DataFrame({"Decision Tree": dt_metrics, "Random Forest": rf_metrics}).T
        comp_df = comp_df.round(4)

        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.dataframe(comp_df.style.format("{:.2%}"), use_container_width=True)
            best_model = "Random Forest" if rf_metrics["Accuracy"] >= dt_metrics["Accuracy"] else "Decision Tree"
            st.markdown(
                f'<div class="insight-box">🏆 النموذج الأفضل في هذه البيانات: <b>{best_model}</b> '
                f'بدقة (Accuracy) = <b>{max(rf_metrics["Accuracy"], dt_metrics["Accuracy"]):.1%}</b></div>',
                unsafe_allow_html=True
            )

        with c2:
            comp_melt = comp_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
            fig = px.bar(
                comp_melt, x="Metric", y="Score", color="index", barmode="group",
                color_discrete_sequence=["#e8825a", "#5b8def"],
                title="مقارنة المقاييس بين النموذجين"
            )
            fig.update_layout(template="plotly_dark", height=380, yaxis_tickformat=".0%", legend_title="النموذج")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎯 جرّب التصنيف بنفسك (Random Forest)")
        cc1, cc2, cc3 = st.columns(3)
        study_in = cc1.slider("ساعات الدراسة اليومية", 0.0, 10.0, 3.0, 0.1)
        usage_in = cc2.slider("ساعات استخدام السوشل ميديا", 0.0, 12.0, 5.0, 0.1)
        unlocks_in = cc3.slider("عدد فتحات الهاتف اليومية", 0, 150, 60, 1)

        stress_reverse = {1: "Low 🟢", 2: "Medium 🟡", 3: "High 🟠", 4: "Very High 🔴"}
        pred_class = rf_model.predict([[study_in, usage_in, unlocks_in]])[0]
        st.markdown(
            f'<div class="metric-card" style="text-align:center;">'
            f'<p style="color:#9aa3b8;">مستوى التوتر المتوقع</p>'
            f'<h1 style="color:#5b8def; margin:0;">{stress_reverse.get(pred_class, pred_class)}</h1>'
            f'</div>', unsafe_allow_html=True
        )
    else:
        st.warning("عدد البيانات المفلترة غير كافٍ لتدريب نماذج التصنيف. وسّع نطاق الفلاتر.")

# ---------------------------------------------------------
# TAB 5: الملخص التنفيذي
# ---------------------------------------------------------
with tab5:
    st.subheader("📝 ملخص تنفيذي بيانات")

    avg_usage = filtered['Avg_Daily_Usage_Hours'].mean()
    avg_sleep = filtered['Sleep_Hours_Per_Night'].mean()
    corr_usage_mh = filtered[['Avg_Daily_Usage_Hours', 'Mental_Health_Score']].corr().iloc[0, 1]
    corr_sleep_mh = filtered[['Sleep_Hours_Per_Night', 'Mental_Health_Score']].corr().iloc[0, 1]
    heavy_pct = (filtered['Student_Category'] == 'Heavy Student').mean() * 100

    st.markdown(f"""
    <div class="insight-box">
    📌 <b>المعطيات الحالية:</b> متوسط الاستخدام اليومي <b>{avg_usage:.1f} ساعة</b>،
    ومتوسط النوم <b>{avg_sleep:.1f} ساعة</b>. نسبة <b>{heavy_pct:.0f}%</b> من الطلاب
    تُصنّف "استخدام مرتفع" (Heavy Student).
    </div>

    <div class="insight-box">
    📌 <b>العلاقة بين الاستخدام والصحة النفسية:</b> معامل ارتباط
    <b>{corr_usage_mh:.2f}</b> — كل زيادة في ساعات الاستخدام ترتبط بانخفاض ملحوظ
    في درجة الصحة النفسية.
    </div>

    <div class="insight-box">
    📌 <b>دور النوم:</b> معامل ارتباط <b>{corr_sleep_mh:.2f}</b> بين النوم والصحة النفسية —
    النوم الكافي أحد أقوى عوامل الحماية النفسية لدى الطلاب.
    </div>

    <div class="warn-box">
    🚨 <b>التوصية:</b> استهداف فئة "Heavy Student" ببرامج توعوية لتقليل وقت الشاشة
    وتحسين جودة النوم، مع متابعة دورية لمستوى التوتر باعتباره مؤشرًا تحذيريًا مبكرًا.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 📋 عرض البيانات الخام (مفلترة)")
    st.dataframe(filtered, use_container_width=True, height=320)
