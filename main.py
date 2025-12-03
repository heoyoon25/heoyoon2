import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats  # T-test용
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel # 변수 선택용
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, mean_absolute_error, 
    mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# 1. 페이지 기본 설정
# ----------------------
st.set_page_config(
    page_title="하이브리드모형 동적 프레임워크（의사결정나무+회귀분석）",
    page_icon="📊",
    layout="wide"
)

# 전역 상태 관리
if "step" not in st.session_state:
    st.session_state.step = 0 
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {
        "imputer": None, "scaler": None, "encoders": None, 
        "feature_cols": None, "target_col": None,
        "feature_candidates": [] 
    }
if "models" not in st.session_state:
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.3, "decision_tree": 0.7}}

# ----------------------
# 2. 사이드바：단계 네비게이션
# ----------------------
st.sidebar.title("📌 하이브리드모형 작업 흐름")
st.sidebar.divider()

steps = ["데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "성능 평가"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

# ----------------------
# 3. 메인 페이지 로직
# ----------------------
st.title("📊 하이브리드모형 동적 배포 프레임워크")
st.divider()

# ==============================================================================
#  단계 0：데이터 업로드
# ==============================================================================
if st.session_state.step == 0:
    st.subheader("📤 데이터 업로드")
    
    tab1, tab2 = st.tabs(["📂 내 파일 업로드", "💾 서버 기본 데이터 사용"])
    
    def load_csv_safe(file_buffer):
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin1']
        for enc in encodings:
            try:
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, encoding=enc)
                return df, enc
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return None, str(e)
        return None, "모든 인코딩 시도 실패"

    with tab1:
        st.markdown("지원 형식：CSV、Parquet、Excel（.xlsx/.xls）")
        uploaded_file = st.file_uploader("데이터 파일 선택", type=["csv", "parquet", "xlsx", "xls"], key="single_file")
        
        if uploaded_file:
            try:
                df = None
                if uploaded_file.name.endswith('.csv'):
                    df, enc_used = load_csv_safe(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if df is not None:
                    df = df.reset_index(drop=True)
                    st.session_state.data["merged"] = df
                    st.success(f"✅ 파일 업로드 성공! ({len(df):,} 행)")
            except Exception as e:
                st.error(f"❌ 파일 처리 중 오류 발생: {e}")
    
    with tab2:
        DEFAULT_FILE_PATH = "Accepted_data (1).csv" 
        if st.button("기본 데이터 불러오기", type="primary"):
            if os.path.exists(DEFAULT_FILE_PATH):
                with open(DEFAULT_FILE_PATH, 'rb') as f:
                    df_default, enc_used = load_csv_safe(f)
                if df_default is not None:
                    st.session_state.data["merged"] = df_default.reset_index(drop=True)
                    st.success(f"✅ 기본 데이터 로드 성공! ({len(df_default):,} 행)")
                    st.rerun()
                else:
                    st.error("❌ 기본 파일을 읽을 수 없습니다.")
            else:
                st.error("⚠️ 파일을 찾을 수 없습니다.")

    if st.session_state.data.get("merged") is not None:
        df_merged = st.session_state.data["merged"]
        st.divider()
        st.markdown(f"### ✅ 현재 로드된 데이터 ({len(df_merged):,} 행)")
        st.dataframe(df_merged.head(5), width='stretch')

# ==============================================================================
#  단계 1：데이터 시각화
# ==============================================================================
elif st.session_state.step == 1:
    st.subheader("📊 데이터 시각화")
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요")
    else:
        df = st.session_state.data["merged"]
        all_cols = df.columns.tolist()
        
        selected_cols = st.multiselect("분석 대상 변수 선택", options=all_cols, default=all_cols[:5])
        
        if selected_cols:
            df_vis = df[selected_cols]
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("📋 X축", ["선택 안 함"] + list(df_vis.columns))
            with col2:
                y_var = st.selectbox("📈 Y축 (수치형 권장)", ["없음"] + list(df_vis.select_dtypes(include=np.number).columns))
            with col3:
                graph_type = st.selectbox("📊 그래프 유형", ["막대 그래프", "박스 플롯", "산점도", "히스토그램"])
            
            st.divider()
            if y_var != "없음" or graph_type == "히스토그램":
                try:
                    if graph_type == "히스토그램":
                        fig = px.histogram(df_vis, x=y_var if y_var!="없음" else x_var, color=x_var if x_var!="선택 안 함" else None)
                    elif graph_type == "막대 그래프" and x_var != "선택 안 함":
                        fig = px.bar(df_vis.groupby(x_var)[y_var].mean().reset_index(), x=x_var, y=y_var)
                    elif graph_type == "박스 플롯" and x_var != "선택 안 함":
                        fig = px.box(df_vis, x=x_var, y=y_var)
                    elif graph_type == "산점도" and x_var != "선택 안 함":
                        fig = px.scatter(df_vis, x=x_var, y=y_var)
                    else:
                        fig = None
                        st.info("축 설정을 확인해주세요.")
                    if fig: st.plotly_chart(fig, use_container_width=True)
                except Exception as e: st.error(f"시각화 오류: {e}")

# ==============================================================================
#  단계 2：데이터 전처리 & 변수 선택
# ==============================================================================
elif st.session_state.step == 2:
    st.subheader("🧹 데이터 전처리 & 변수 선택")
    
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요.")
    else:
        df_origin = st.session_state.data["merged"].copy()
        all_cols = df_origin.columns.tolist()

        # ---------------------------------------------------------
        # 0. 타겟 변수 선택
        # ---------------------------------------------------------
        st.markdown("### 0️⃣ 타겟 변수 설정")
        default_idx = all_cols.index("Loan_status") if "Loan_status" in all_cols else 0
        target_col = st.selectbox("🎯 타겟 변수 (Y) 선택", options=all_cols, index=default_idx)
        st.session_state.preprocess["target_col"] = target_col
        st.divider()

        # ---------------------------------------------------------
        # 1. 데이터 전처리 실행
        # ---------------------------------------------------------
        st.markdown("### 1️⃣ 데이터 전처리 실행")
        st.info("💡 **수행 작업**: 결측치 40% 이상 제거 / 단일값 제거 / 최빈값 99% 이상 제거 / 범주 100개 이상 제거 / 결측치 대치 / 스케일링 / 인코딩")

        if st.button("🚀 전처리 및 정제 시작", type="primary"):
            with st.spinner("데이터 정제 및 변환 중..."):
                try:
                    # 1) 타겟 결측 제거
                    clean_df = df_origin.dropna(subset=[target_col]).reset_index(drop=True)
                    
                    # 2) X 분리
                    X_raw = clean_df.drop(columns=[target_col])
                    y = clean_df[target_col].copy()

                    # 3) 삭제 로직 (강화된 기준 적용)
                    drop_cols = []
                    
                    # A. 결측치 40% 이상 삭제
                    missing_ratio = X_raw.isna().mean()
                    high_missing = missing_ratio[missing_ratio >= 0.40].index.tolist()
                    drop_cols.extend(high_missing)

                    # B. 단일 값(상수) 삭제 / C. 최빈값 99% 이상 삭제
                    for col in X_raw.columns:
                        if col in drop_cols: continue
                        
                        # 단일 값
                        if X_raw[col].nunique() <= 1:
                            drop_cols.append(col)
                            continue
                        
                        # 최빈값 99% 이상
                        most_freq_ratio = X_raw[col].value_counts(normalize=True).iloc[0]
                        if most_freq_ratio >= 0.99:
                            drop_cols.append(col)

                    # D. 범주 수가 100개 이상인 범주형 변수 삭제
                    cat_cols_raw = X_raw.select_dtypes(include=['object', 'category']).columns
                    high_cardinality = [c for c in cat_cols_raw if X_raw[c].nunique() >= 100]
                    drop_cols.extend(high_cardinality)

                    # 중복 제거 후 삭제 실행
                    drop_cols = list(set(drop_cols))
                    X_raw = X_raw.drop(columns=drop_cols)
                    
                    if drop_cols:
                        st.warning(f"⚠️ 총 {len(drop_cols)}개 변수가 기준(결측 40%↑, 빈도 99%↑, 단일값, 범주 100개↑)에 의해 제거되었습니다.")
                        with st.expander("제거된 변수 목록 보기"):
                            st.write(drop_cols)

                    # 4) 타겟 인코딩
                    le_target = None
                    if y.dtype == 'object' or y.dtype.name == 'category':
                        le_target = LabelEncoder()
                        y = pd.Series(le_target.fit_transform(y), index=y.index)

                    # 5) 결측치 대치 / 스케일링 / 인코딩
                    X = X_raw.copy()
                    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

                    imputer = SimpleImputer(strategy='mean')
                    scaler = StandardScaler()
                    encoders = {}

                    # 수치형
                    if num_cols:
                        X[num_cols] = imputer.fit_transform(X[num_cols])
                        X[num_cols] = scaler.fit_transform(X[num_cols])
                    
                    # 범주형
                    for col in cat_cols:
                        X[col] = X[col].fillna("Unknown").astype(str)
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le
                    
                    final_features = list(X.columns)
                    
                    # 세션 저장
                    st.session_state.data["X_candidates"] = X
                    st.session_state.data["y_processed"] = y
                    st.session_state.preprocess["feature_candidates"] = final_features
                    st.session_state.preprocess["target_encoder"] = le_target
                    
                    st.success(f"✅ 기본 전처리 완료! (남은 변수: {len(final_features)}개)")
                    st.dataframe(X.head())

                except Exception as e:
                    st.error(f"전처리 중 오류: {e}")

        st.divider()

        # ---------------------------------------------------------
        # 2. T-test (이진 분류용)
        # ---------------------------------------------------------
        st.markdown("### 2️⃣ T-test (통계적 가설 검정)")
        
        if "X_candidates" in st.session_state.data:
            X_curr = st.session_state.data["X_candidates"]
            y_curr = st.session_state.data["y_processed"]
            
            unique_y = np.unique(y_curr)
            if len(unique_y) == 2:
                if st.button("🧪 T-test 실행 (p-value < 0.05 변수 선택)"):
                    with st.spinner("T-test 수행 중..."):
                        selected_by_ttest = []
                        p_values = {}

                        group0_idx = (y_curr == unique_y[0])
                        group1_idx = (y_curr == unique_y[1])

                        for col in X_curr.columns:
                            try:
                                val0 = X_curr.loc[group0_idx, col]
                                val1 = X_curr.loc[group1_idx, col]
                                
                                stat, p_val = stats.ttest_ind(val0, val1, equal_var=False)
                                
                                if p_val < 0.05:
                                    selected_by_ttest.append(col)
                                    p_values[col] = p_val
                            except:
                                continue
                        
                        if selected_by_ttest:
                            st.session_state.preprocess["feature_candidates"] = selected_by_ttest
                            st.session_state.data["X_candidates"] = X_curr[selected_by_ttest]
                            st.success(f"✅ T-test 완료! 유의미한 변수 {len(selected_by_ttest)}개가 선택되었습니다.")
                            
                            res_df = pd.DataFrame({"Variable": selected_by_ttest, "P-value": [p_values[c] for c in selected_by_ttest]})
                            st.dataframe(res_df.sort_values("P-value"), height=200)
                        else:
                            st.warning("⚠️ 유의미한 변수(p<0.05)가 하나도 없습니다.")
            else:
                st.info("ℹ️ T-test는 타겟 변수가 이진 분류(클래스 2개)일 때만 활성화됩니다.")
        else:
            st.info("먼저 1번 전처리를 실행해주세요.")

        st.divider()

        # ---------------------------------------------------------
        # 3. 최종 변수 확정
        # ---------------------------------------------------------
        st.markdown("### 3️⃣ 최종 입력 변수(X) 확인 및 확정")
        if "X_candidates" in st.session_state.data:
            current_candidates = st.session_state.preprocess["feature_candidates"]
            
            selected_features = st.multiselect(
                "최종 모델에 사용할 변수를 확정하세요:",
                options=current_candidates,
                default=current_candidates,
                key="final_multiselect"
            )
            
            if st.button("✅ 변수 확정 및 다음 단계로"):
                if not selected_features:
                    st.error("최소 1개 이상의 변수를 선택해야 합니다.")
                else:
                    st.session_state.data["X_processed"] = st.session_state.data["X_candidates"][selected_features]
                    st.session_state.preprocess["feature_cols"] = selected_features
                    st.success(f"최종 {len(selected_features)}개 변수가 확정되었습니다. '모델 학습' 탭으로 이동하세요!")

# ==============================================================================
#  단계 3：모델 학습
# ==============================================================================
elif st.session_state.step == 3:
    st.subheader("🚀 모델 학습 설정")

    if "X_processed" not in st.session_state.data:
        st.warning("⚠ 먼저 전처리 단계에서 변수를 확정하세요.")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]

        # 1. 분석 유형
        task_option = st.radio("분석 유형:", ["분류 (Classification)", "회귀 (Regression)"], horizontal=True)
        is_classification = "분류" in task_option
        st.session_state["is_classification"] = is_classification

        st.divider()

        if "selected_logit_features" not in st.session_state:
            st.session_state.selected_logit_features = list(X.columns)
        if "selected_tree_features" not in st.session_state:
            st.session_state.selected_tree_features = list(X.columns)

        col_conf1, col_conf2 = st.columns(2)

        # -------------------------------------------------------------
        # A. Logit / Stepwise 설정 (속도 개선 적용됨)
        # -------------------------------------------------------------
        with col_conf1:
            st.markdown("#### 🔹 Logit / Linear & Stepwise")
            with st.expander("설정 열기", expanded=True):
                # Stepwise 버튼
                if st.button("Stepwise 변수 선택 (Auto)", help="속도를 위해 데이터 일부를 샘플링하여 변수를 선택합니다."):
                    with st.spinner("Stepwise(Forward) 진행 중... (데이터 양에 따라 시간이 걸릴 수 있습니다)"):
                        try
