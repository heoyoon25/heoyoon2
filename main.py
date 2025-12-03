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
                        try:
                            # 1. 속도 개선을 위한 샘플링
                            if len(X) > 2000:
                                X_sample = X.sample(n=2000, random_state=42)
                                y_sample = y.loc[X_sample.index]
                                st.caption("🚀 속도 향상을 위해 2,000개의 표본 데이터로 변수를 선택했습니다.")
                            else:
                                X_sample = X
                                y_sample = y
                            
                            # 2. 모델 및 SFS 설정 (cv=3, n_jobs=-1)
                            est = LogisticRegression(solver='lbfgs', max_iter=200) if is_classification else LinearRegression()
                            
                            sfs = SequentialFeatureSelector(
                                est, 
                                n_features_to_select='auto', 
                                direction='forward',
                                cv=3,  # 교차검증 횟수 단축
                                n_jobs=-1 # 병렬 처리
                            )
                            
                            sfs.fit(X_sample, y_sample)
                            
                            selected_mask = sfs.get_support()
                            selected_features = X.columns[selected_mask].tolist()
                            
                            if not selected_features:
                                st.warning("선택된 변수가 없습니다.")
                            else:
                                st.session_state.selected_logit_features = selected_features
                                st.success(f"Stepwise 완료! {len(selected_features)}개 변수 선택됨.")
                                
                        except Exception as e:
                            st.error(f"Stepwise 오류: {e}")

                final_logit_feats = st.multiselect(
                    "Logit 모델 사용 변수", 
                    options=list(X.columns),
                    default=st.session_state.selected_logit_features,
                    key="logit_feats_select"
                )
                
                test_size_logit = st.slider("Test 비율 (Logit)", 0.1, 0.4, 0.2)
                if is_classification:
                    C_logit = st.slider("규제 강도(C)", 0.01, 10.0, 1.0)

        # -------------------------------------------------------------
        # B. Tree / CART 설정
        # -------------------------------------------------------------
        with col_conf2:
            st.markdown("#### 🌳 Decision Tree (CART)")
            with st.expander("설정 열기", expanded=True):
                # CART Selection 버튼
                if st.button("Decision Tree(CART) 변수 선택 (Auto)", help="트리 중요도(Feature Importance) 기반 상위 변수 선택"):
                    with st.spinner("CART 변수 중요도 분석 중..."):
                        try:
                            est_tree = DecisionTreeClassifier(random_state=42) if is_classification else DecisionTreeRegressor(random_state=42)
                            est_tree.fit(X, y)
                            
                            selector = SelectFromModel(est_tree, prefit=True)
                            selected_mask_tree = selector.get_support()
                            
                            st.session_state.selected_tree_features = X.columns[selected_mask_tree].tolist()
                            st.success(f"CART 선택 완료! {sum(selected_mask_tree)}개 변수 선택됨.")
                        except Exception as e:
                            st.error(f"Tree 선택 오류: {e}")

                final_tree_feats = st.multiselect(
                    "Tree 모델 사용 변수", 
                    options=list(X.columns),
                    default=st.session_state.selected_tree_features,
                    key="tree_feats_select"
                )

                test_size_tree = st.slider("Test 비율 (Tree)", 0.1, 0.4, 0.2)
                tree_depth = st.slider("Max Depth", 2, 20, 6)

        st.divider()
        st.markdown("#### ⚖ Hybrid 가중치")
        reg_weight = st.slider("Logit 가중치 (나머지는 Tree)", 0.0, 1.0, 0.5)

        # -------------------------------------------------------------
        # 학습 시작
        # -------------------------------------------------------------
        if st.button("🏁 모델 학습 시작 (최종 선택 변수 적용)", type="primary"):
            try:
                # 1. Logit 데이터셋 준비
                X_logit = X[final_logit_feats]
                X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
                    X_logit, y, test_size=test_size_logit, random_state=42, stratify=y if is_classification else None
                )
                
                # 2. Tree 데이터셋 준비
                X_tree = X[final_tree_feats]
                X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
                    X_tree, y, test_size=test_size_tree, random_state=42, stratify=y if is_classification else None
                )
                
                # 3. 모델 정의 및 학습
                if is_classification:
                    model_l = LogisticRegression(C=C_logit, max_iter=1000)
                    model_t = DecisionTreeClassifier(max_depth=tree_depth, random_state=42)
                else:
                    model_l = LinearRegression()
                    model_t = DecisionTreeRegressor(max_depth=tree_depth, random_state=42)

                model_l.fit(X_train_l, y_train_l)
                model_t.fit(X_train_t, y_train_t)

                st.session_state.models["logit_model"] = model_l
                st.session_state.models["tree_model"] = model_t
                st.session_state.models["hybrid_weight"] = reg_weight
                
                # 평가용 데이터 저장 (Logit split 기준)
                st.session_state.data["eval_set"] = {
                    "y_test": y_test_l,
                    "X_test_logit": X_test_l,
                    "X_test_tree": X_tree.loc[X_test_l.index] 
                }

                st.success("학습 완료! 성능 평가 페이지로 이동하세요.")
            except Exception as e:
                st.error(f"학습 중 오류: {e}")

# ==============================================================================
#  단계 4：성능 평가
# ==============================================================================
elif st.session_state.step == 4:
    st.subheader("📈 모델 성능 심층 평가")

    if "eval_set" not in st.session_state.data:
        st.warning("⚠️ 모델 학습을 먼저 완료하세요.")
    else:
        # 데이터 로드
        eval_data = st.session_state.data["eval_set"]
        y_test = eval_data["y_test"]
        X_test_l = eval_data["X_test_logit"]
        X_test_t = eval_data["X_test_tree"]
        
        model_l = st.session_state.models["logit_model"]
        model_t = st.session_state.models["tree_model"]
        w = st.session_state.models["hybrid_weight"]
        is_cls = st.session_state.get("is_classification", True)

        if is_cls:
            # 분류 평가
            prob_l = model_l.predict_proba(X_test_l)[:, 1]
            prob_t = model_t.predict_proba(X_test_t)[:, 1]
            prob_h = w * prob_l + (1-w) * prob_t
            pred_h = (prob_h >= 0.5).astype(int)
            pred_l = model_l.predict(X_test_l)
            pred_t = model_t.predict(X_test_t)

            def get_metrics(y_true, y_pred, y_prob):
                return {
                    "Acc": accuracy_score(y_true, y_pred),
                    "F1": f1_score(y_true, y_pred, zero_division=0),
                    "AUC": auc(*roc_curve(y_true, y_prob)[:2])
                }
            
            m1 = get_metrics(y_test, pred_l, prob_l)
            m2 = get_metrics(y_test, pred_t, prob_t)
            m3 = get_metrics(y_test, pred_h, prob_h)

            st.table(pd.DataFrame([m1, m2, m3], index=["Logit", "Tree", "Hybrid"]))
            
            # ROC Curve
            fpr_h, tpr_h, _ = roc_curve(y_test, prob_h)
            fig = px.area(x=fpr_h, y=tpr_h, title="Hybrid ROC Curve", labels=dict(x="FPR", y="TPR"))
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig)

            # Confusion Matrix
            cm = confusion_matrix(y_test, pred_h)
            fig_cm = px.imshow(cm, text_auto=True, title="Hybrid Confusion Matrix", color_continuous_scale='Blues')
            st.plotly_chart(fig_cm)

        else:
            # 회귀 평가
            pred_l = model_l.predict(X_test_l)
            pred_t = model_t.predict(X_test_t)
            pred_h = w * pred_l + (1-w) * pred_t
            
            mae = mean_absolute_error(y_test, pred_h)
            r2 = r2_score(y_test, pred_h)
            st.metric("Hybrid MAE", f"{mae:.4f}")
            st.metric("Hybrid R2", f"{r2:.4f}")
            
            fig = px.scatter(x=y_test, y=pred_h, labels={'x':'Actual', 'y':'Predicted'}, title="Actual vs Predicted")
            fig.add_shape(type='line', line=dict(dash='dash', color='red'), x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max())
            st.plotly_chart(fig)
