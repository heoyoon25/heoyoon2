import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, mean_absolute_error, 
    mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 0. 전처리 함수 (전역으로 분리하여 재사용성 확보)
# 데이터 누수 방지를 위해 X_train_raw 데이터로만 fit하고 X_test_raw에 transform을 적용합니다.
# ==============================================================================
def preprocess_data_for_model(X_train_raw, X_test_raw, num_cols, cat_cols, is_logit=True):
    """
    각 모델의 특성에 맞게 데이터를 전처리합니다. (Logit: OHE+Scaling, Tree: LE+Imputation)
    """
    # 훈련/테스트 데이터 복사 및 결측치 초기 대치 (범주형/수치형 모두 일단 'Unknown'으로)
    X_train = X_train_raw.copy().fillna('Unknown') 
    X_test = X_test_raw.copy().fillna('Unknown')
    
    # 1. 수치형 처리 (Imputation + Scaling)
    if num_cols:
        # 'Unknown'을 다시 np.nan으로 변경하여 Imputer가 처리하도록 함
        X_train_num_imputed_df = X_train[num_cols].replace('Unknown', np.nan)
        X_test_num_imputed_df = X_test[num_cols].replace('Unknown', np.nan)
        
        imputer = SimpleImputer(strategy='mean')
        
        # 훈련 데이터에 fit 후 transform
        X_train_num_imputed = imputer.fit_transform(X_train_num_imputed_df)
        X_test_num_imputed = imputer.transform(X_test_num_imputed_df)
        
        # Logit 모델일 경우 스케일링
        if is_logit:
            scaler = StandardScaler()
            X_train_num_scaled = scaler.fit_transform(X_train_num_imputed)
            X_test_num_scaled = scaler.transform(X_test_num_imputed)
        else:
            X_train_num_scaled = X_train_num_imputed # Tree 모델은 스케일링 불필요
            X_test_num_scaled = X_test_num_imputed
            
        X_train_num = pd.DataFrame(X_train_num_scaled, columns=num_cols, index=X_train.index)
        X_test_num = pd.DataFrame(X_test_num_scaled, columns=num_cols, index=X_test.index)
    else:
        X_train_num = pd.DataFrame(index=X_train.index)
        X_test_num = pd.DataFrame(index=X_test.index)
        
    # 2. 범주형 처리 (Logit: OHE, Tree: Label Encoding)
    if cat_cols:
        if is_logit:
            # Logit: One-Hot Encoding
            X_train_cat = pd.get_dummies(X_train[cat_cols].astype(str), prefix=cat_cols)
            X_test_cat = pd.get_dummies(X_test[cat_cols].astype(str), prefix=cat_cols)
            
            # 훈련/테스트셋 컬럼 일치 (OHE 필수)
            train_cols = X_train_cat.columns
            X_test_cat = X_test_cat.reindex(columns=train_cols, fill_value=0)
            
        else:
            # Tree: Label Encoding (결측치는 'Unknown'으로 이미 처리됨)
            X_train_cat = pd.DataFrame(index=X_train.index)
            X_test_cat = pd.DataFrame(index=X_test.index)
            for col in cat_cols:
                le = LabelEncoder()
                le.fit(X_train[col].astype(str)) # Tree 훈련 데이터에 fit
                
                # 테스트셋에 없는 레이블은 무시 (오류 방지)
                test_labels = np.array(X_test[col].astype(str))
                train_classes = set(le.classes_)
                
                # 훈련셋에 없는 값은 새로운 레이블 (-1) 부여
                test_mapped = np.array([le.transform([x])[0] if x in train_classes else -1 
                                        for x in test_labels])
                X_train_cat[col] = le.transform(X_train[col].astype(str))
                X_test_cat[col] = test_mapped
    else:
        X_train_cat = pd.DataFrame(index=X_train.index)
        X_test_cat = pd.DataFrame(index=X_test.index)
    
    # 3. 최종 병합
    X_train_processed = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_processed = pd.concat([X_test_num, X_test_cat], axis=1)

    # 최종 정리 (무한대/잔여 결측치 0으로 대치)
    X_train_processed = X_train_processed.replace([np.inf, -np.inf], 0).fillna(0)
    X_test_processed = X_test_processed.replace([np.inf, -np.inf], 0).fillna(0)
    
    return X_train_processed, X_test_processed


# ----------------------
# 1. 페이지 기본 설정
# ----------------------
st.set_page_config(
    page_title="하이브리드 모델 구축 및 비교",
    page_icon="📊",
    layout="wide"
)

# 전역 상태 관리
if "step" not in st.session_state:
    st.session_state.step = 0
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}
if "preprocess" not in st.session_state:
    # 전처리 기준 정보 (fit된 객체가 아닌, 컬럼 목록을 저장)
    st.session_state.preprocess = {
        "num_cols": [], 
        "cat_cols": [], 
        "target_col": None,
        "target_encoder": None
    }
if "models" not in st.session_state:
    st.session_state.models = {
        "regression": None,
        "decision_tree": None,
        "test_size_reg": 0.2, 
        "test_size_dt": 0.2, 
        "mixed_weights": {"regression": 0.5, "decision_tree": 0.5}
    }
if "task" not in st.session_state:
    st.session_state.task = "logit"
    

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
# 3. 메인 페이지：단계별 내용 표시
# ----------------------
st.title("📊 하이브리드 모델 구축 및 비교")
st.divider()

# ==============================================================================
# 메인 로직 시작
# ==============================================================================

# ----------------------
#  단계 0：데이터 업로드
# ----------------------
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
                    if df is None:
                        st.error(f"❌ CSV 파일 읽기 실패: {enc_used}")
                    else:
                        st.caption(f"ℹ️ 감지된 인코딩: {enc_used}")
                        
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
        DEFAULT_FILE_PATH = "accepted_data.csv"
        st.info(f"💡 **기본 데이터 설명**: 대출 관련 통합 데이터 (`{DEFAULT_FILE_PATH}`)")
        
        if st.button("기본 데이터 불러오기", type="primary"):
            if os.path.exists(DEFAULT_FILE_PATH):
                try:
                    with open(DEFAULT_FILE_PATH, 'rb') as f:
                        df_default, enc_used = load_csv_safe(f)
                    
                    if df_default is not None:
                        st.session_state.data["merged"] = df_default.reset_index(drop=True)
                        st.success(f"✅ 기본 데이터 로드 성공! ({len(df_default):,} 행, 인코딩: {enc_used})")
                        st.rerun()
                    else:
                        st.error("❌ 기본 파일을 읽을 수 없습니다 (인코딩 오류).")
                except Exception as e:
                    st.error(f"❌ 기본 파일 로드 중 오류 발생: {e}")
            else:
                st.error(f"⚠️ 파일을 찾을 수 없습니다: {DEFAULT_FILE_PATH}")

    if st.session_state.data.get("merged") is not None:
        df_merged = st.session_state.data["merged"]
        st.divider()
        st.markdown(f"### ✅ 현재 로드된 데이터 ({len(df_merged):,} 행)")
        st.dataframe(df_merged.head(5), width='stretch')

# ----------------------
#  단계 1：데이터 시각화
# ----------------------
elif st.session_state.step == 1:
    st.subheader("📊 데이터 시각화")
    
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요")
    else:
        df = st.session_state.data["merged"]
        
        st.markdown("### 1️⃣ 시각화할 변수 선택")
        all_cols = df.columns.tolist()
        default_selection = all_cols[:10] if len(all_cols) > 10 else all_cols
        
        selected_cols = st.multiselect(
            "분석 대상 변수 선택",
            options=all_cols,
            default=default_selection
        )
        
        if not selected_cols:
            st.error("⚠️ 최소 하나 이상의 변수를 선택해야 시각화가 가능합니다.")
        else:
            df_vis = df[selected_cols]
            st.divider()
            
            st.markdown("### 2️⃣ 그래프 설정")
            cat_cols = df_vis.select_dtypes(include=["object", "category"]).columns.tolist()
            num_cols = df_vis.select_dtypes(include=["int64", "float64"]).columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("📋 X축 (범주형)", ["선택 안 함"] + cat_cols)
                if x_var == "선택 안 함": x_var = None
            with col2:
                y_var = st.selectbox("📈 Y축 (수치형)", num_cols if num_cols else ["없음"])
            with col3:
                graph_type = st.selectbox("📊 그래프 유형", [
                    "막대 그래프", "박스 플롯", "산점도", "히스토그램", "선 그래프"
                ])
            
            st.divider()
            
            if y_var and y_var != "없음":
                try:
                    if graph_type == "히스토그램":
                        fig = px.histogram(df_vis, x=y_var, color=x_var, title=f"{y_var} 분포")
                    elif graph_type == "막대 그래프" and x_var:
                        avg_df = df_vis.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.bar(avg_df, x=x_var, y=y_var, color=x_var, title=f"{x_var}별 {y_var} 평균")
                    elif graph_type == "박스 플롯" and x_var:
                        fig = px.box(df_vis, x=x_var, y=y_var, color=x_var, title=f"{x_var}별 {y_var} 분포")
                    elif graph_type == "산점도" and x_var:
                        fig = px.scatter(df_vis, x=x_var, y=y_var, color=x_var, title=f"{x_var} vs {y_var}")
                    elif graph_type == "선 그래프" and x_var:
                        line_df = df_vis.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.line(line_df, x=x_var, y=y_var, markers=True, title=f"{x_var}별 {y_var} 추세")
                    else:
                        fig = None
                        st.info("X축 변수를 선택해주세요.")
                        
                    if fig:
                        st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"그래프 생성 오류: {e}")
            else:
                st.info("Y축 변수를 선택하면 그래프가 표시됩니다.")

# ----------------------
#  단계 2：데이터 전처리 (변수 분류 및 Y-타겟 처리만 수행)
# ----------------------
elif st.session_state.step == 2:
    st.subheader("🧹 데이터 전처리 & 변수 선택")
    
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요.")
    else:
        df_origin = st.session_state.data["merged"].copy()
        all_cols = df_origin.columns.tolist()

        st.markdown("### 1️⃣ 분석 변수 설정")
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox(
                "🎯 타겟 변수 (Y) 선택", 
                options=all_cols,
                help="예측하고자 하는 목표 변수입니다."
            )
            
        feature_candidates = [c for c in all_cols if c != target_col]
        
        with col2:
            default_feats = feature_candidates[:10] if len(feature_candidates) > 10 else feature_candidates
            selected_features = st.multiselect(
                "📋 입력 변수 (X) 선택",
                options=feature_candidates,
                default=default_feats,
                help="타겟 변수를 예측하기 위해 사용할 데이터입니다."
            )
        
        st.divider()

        if not selected_features:
            st.error("⚠️ 분석할 변수를 선택해주세요.")
        else:
            
            tabs = st.tabs(["⚡ 전처리 실행"])
            tab1 = tabs[0]
            
            with tab1:
                st.write(f"**Y(타겟) 결측치 제거** 및 **X 변수 목록 분류**를 수행합니다.")
                st.caption("ℹ️ X 변수의 실제 스케일링/결측치 처리는 **데이터 누수 방지**를 위해 **'모델 학습' 단계**에서 진행됩니다.")
                
                if st.button("🚀 데이터 정제 및 변수 분류 시작", type="primary"):
                    with st.spinner("데이터 정제 및 변수 분류 중..."):
                        try:
                            if target_col in selected_features:
                                selected_features.remove(target_col)
                                
                            # 1. 타겟(Y) 결측치 처리
                            clean_df = df_origin.dropna(subset=[target_col]).reset_index(drop=True)
                            dropped_count = len(df_origin) - len(clean_df)
                            if dropped_count > 0:
                                st.warning(f"⚠️ 타겟 변수({target_col})가 비어있는 {dropped_count}개 행을 제거했습니다.")
                            
                            X_raw = clean_df[selected_features].copy()
                            y = clean_df[target_col].copy()
                            
                            # 2. 타겟 변수(Y) 인코딩 처리
                            le_target = None
                            if y.dtype == 'object' or y.dtype.name == 'category':
                                le_target = LabelEncoder()
                                y = pd.Series(le_target.fit_transform(y), index=y.index)
                                st.info(f"ℹ️ 타겟 변수 '{target_col}'가 문자열 형식이어서 숫자로 변환(Label Encoding)했습니다.")
                                mapping_info = {i: label for i, label in enumerate(le_target.classes_)}
                                st.caption(f"└ 변환 정보: {mapping_info}")

                            # 3. 입력 변수(X) 분류 (실제 변환은 단계 3에서)
                            num_cols = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
                            
                            # 4. 전역 상태(Session State)에 저장
                            st.session_state.preprocess.update({
                                "target_col": target_col,
                                "target_encoder": le_target,
                                "num_cols": num_cols,
                                "cat_cols": cat_cols,
                            })
                            st.session_state.data["X_raw"] = X_raw
                            st.session_state.data["y_processed"] = y
                            
                            st.success(f"✅ 변수 분류 완료! (수치형: {len(num_cols)}개, 범주형: {len(cat_cols)}개, 데이터: {len(X_raw)}행)")
                            st.dataframe(X_raw.head(), width='stretch')
                            
                        except Exception as e:
                            st.error(f"❌ 전처리 중 오류 발생: {str(e)}")
                else:
                    st.info("👈 위 버튼을 눌러 전처리를 시작하세요.")

# ==============================================================================
#  단계 3：모델 학습 (데이터 분할 및 전처리 동시 수행)
# ==============================================================================
elif st.session_state.step == 3:
    st.subheader("🚀 모델 학습 설정")
    
    if "X_raw" not in st.session_state.data:
        st.warning("⚠️ 먼저 [데이터 전처리] 단계를 완료하세요.")
    else:
        # X_raw와 y_processed 로드
        X_raw = st.session_state.data["X_raw"]
        y = st.session_state.data["y_processed"]
        num_cols = st.session_state.preprocess["num_cols"]
        cat_cols = st.session_state.preprocess["cat_cols"]
        
        # -------------------------------------------------------------
        # 1. 분석 유형 선택
        # -------------------------------------------------------------
        st.markdown("### 1️⃣ 분석 유형 선택")
        task_option = st.radio(
            "데이터의 타겟(Y) 특성에 맞는 유형을 선택하세요:",
            ["분류 (Classification)", "회귀 (Regression)"],
            horizontal=True,
            key="task_radio"
        )
        st.session_state.task = "logit" if "분류" in task_option else "tree"
        
        st.divider()

        # -------------------------------------------------------------
        # 2. 모델 설정 및 데이터 분할 (개별 분할 설정)
        # -------------------------------------------------------------
        st.markdown("### 2️⃣ 모델 설정 및 데이터 분할")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 🔹 Logit 모델 (회귀/로지스틱)")
            test_size_reg = st.slider(
                "Logit 테스트 데이터 비율", 
                0.1, 0.4, st.session_state.models["test_size_reg"], 0.05, 
                key="test_size_reg",
                help="Logit 모델 학습 시 사용할 테스트 데이터의 비율입니다."
            )
            st.session_state.models["test_size_reg"] = test_size_reg
            st.caption("🔧 **전처리**: OHE(범주형), StandardScaler(수치형)")

        with col2:
            st.markdown("##### 🌳 Tree 모델 (의사결정나무)")
            test_size_dt = st.slider(
                "Tree 테스트 데이터 비율", 
                0.1, 0.4, st.session_state.models["test_size_dt"], 0.05, 
                key="test_size_dt",
                help="Tree 모델 학습 시 사용할 테스트 데이터의 비율입니다."
            )
            st.session_state.models["test_size_dt"] = test_size_dt
            
            tree_depth = st.slider("최대 깊이 (Max Depth)", 1, 20, 5, key="tree_depth")
            st.caption(f"깊이 제한: {tree_depth}")
            st.caption("🔧 **전처리**: Label Encoding(범주형), Imputation(수치형)")

        with col3:
            st.markdown("##### ⚖️ Hybrid 모델 (결합 모형)")
            st.caption("Logit + Tree 예측 결과 가중치")
            reg_weight = st.slider("Logit 가중치", 0.0, 1.0, st.session_state.models["mixed_weights"]["regression"], 0.1, key="reg_weight")
            st.session_state.models["mixed_weights"]["regression"] = reg_weight
            st.session_state.models["mixed_weights"]["decision_tree"] = 1.0 - reg_weight
            st.caption(f"비율: Logit {int(reg_weight*100)}% : Tree {int((1-reg_weight)*100)}%")

        st.divider()
        
        # -------------------------------------------------------------
        # 3. 학습 시작 버튼
        # -------------------------------------------------------------
        if st.button("🏁 모델 학습 시작", type="primary"):
            with st.spinner("3가지 모델을 모두 학습 중입니다..."):
                try:
                    # 1. Logit 모델용 데이터 분할
                    stratify_reg = y if st.session_state.task == "logit" and y.nunique() > 1 else None
                    X_train_raw_reg, X_test_raw_reg, y_train_reg, y_test_reg = train_test_split(
                        X_raw, y, test_size=test_size_reg, random_state=42, stratify=stratify_reg
                    )
                    
                    # 2. Logit 모델용 데이터 전처리 (OHE + Scaling)
                    # 전역 함수 사용
                    X_train_reg, X_test_reg = preprocess_data_for_model(
                        X_train_raw_reg, X_test_raw_reg, num_cols, cat_cols, is_logit=True
                    )
                    
                    # 3. Logit 모델 학습
                    if st.session_state.task == "logit":
                        reg_model = LogisticRegression(max_iter=5000, random_state=42, solver='liblinear') # Max_iter 증가, solver 명시
                    else:
                        reg_model = LinearRegression()
                    reg_model.fit(X_train_reg, y_train_reg)


                    # 4. Tree 모델용 데이터 분할
                    stratify_dt = y if st.session_state.task == "logit" and y.nunique() > 1 else None
                    X_train_raw_dt, X_test_raw_dt, y_train_dt, y_test_dt = train_test_split(
                        X_raw, y, test_size=test_size_dt, random_state=42, stratify=stratify_dt
                    )

                    # 5. Tree 모델용 데이터 전처리 (Label Encoding + Imputation)
                    # 전역 함수 사용
                    X_train_dt, X_test_dt = preprocess_data_for_model(
                        X_train_raw_dt, X_test_raw_dt, num_cols, cat_cols, is_logit=False
                    )

                    # 6. Tree 모델 학습
                    if st.session_state.task == "logit":
                        dt_model = DecisionTreeClassifier(max_depth=tree_depth, random_state=42)
                    else:
                        dt_model = DecisionTreeRegressor(max_depth=tree_depth, random_state=42)
                    dt_model.fit(X_train_dt, y_train_dt)
                    
                    # 7. 결과 저장
                    st.session_state.models["regression"] = reg_model
                    st.session_state.models["decision_tree"] = dt_model
                    
                    # 전처리된 테스트셋 저장 (평가에 사용)
                    st.session_state.data.update({
                        "X_test_reg": X_test_reg, "y_test_reg": y_test_reg,
                        "X_test_dt": X_test_dt, "y_test_dt": y_test_dt,
                        "X_test_raw_reg": X_test_raw_reg, # Hybrid 평가를 위한 Logit 테스트셋 원본
                        "X_train_raw_dt": X_train_raw_dt # Hybrid 평가를 위한 Tree 훈련셋 원본 (새로 추가)
                    })

                    st.success("✅ 모든 모델의 학습이 완료되었습니다!")
                    st.info(f"👉 **'성능 평가' 단계로 이동하여 3개 모델의 성능을 비교하세요.**"
                             f"\n\n**Logit 모델**: {test_size_reg*100:.0f}% 테스트셋 사용 (OHE/Scaling 적용)"
                             f"\n**Tree 모델**: {test_size_dt*100:.0f}% 테스트셋 사용 (Label Encoding 적용)")
                    
                    st.button("👉 성능 평가 단계로 이동", on_click=lambda: st.session_state.update(step=4))

                except Exception as e:
                    st.error(f"학습 중 오류 발생: {e}")
                            
# ==============================================================================
#  단계 4：성능 평가 
# (중복 함수 제거 및 전역 함수를 사용하도록 로직 개선)
# ==============================================================================
elif st.session_state.step == 4:
    st.subheader("📈 모델 성능 심층 평가")
    
    if st.session_state.models["regression"] is None:
        st.warning("⚠️ 먼저 [모델 학습] 단계를 완료하세요")
    else:
        # 데이터 및 모델 로드
        reg_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        w = st.session_state.models["mixed_weights"]
        num_cols = st.session_state.preprocess["num_cols"]
        cat_cols = st.session_state.preprocess["cat_cols"]
        
        X_test_reg = st.session_state.data["X_test_reg"]
        y_test_reg = st.session_state.data["y_test_reg"]
        X_test_dt = st.session_state.data["X_test_dt"]
        y_test_dt = st.session_state.data["y_test_dt"]
        X_test_raw_reg = st.session_state.data["X_test_raw_reg"]
        # Tree 훈련셋 원본 데이터 로드 (Logit 테스트셋을 전처리할 때의 기준)
        X_train_raw_dt = st.session_state.data["X_train_raw_dt"] 

        st.info(f"ℹ️ Hybrid 가중치: Logit {w['regression']*100:.0f}% + Tree {w['decision_tree']*100:.0f}%"
                f" (평가는 Logit 모델의 테스트셋 크기({len(X_test_reg)}행)를 기준으로 진행됩니다.)")

        # ----------------------------------------------------------------------
        # A. Tree 모델의 Logit 테스트셋 예측값을 얻기 위한 재전처리 
        # (중복 함수 제거: get_tree_preds_on_logit_test 대신 전역 함수 preprocess_data_for_model 사용)
        # ----------------------------------------------------------------------
        
        # Logit 테스트셋(raw)을 Tree 모델의 훈련셋 기준으로 전처리하여 Tree 모델이 예측할 수 있게 준비
        _, X_test_for_tree_on_logit_set = preprocess_data_for_model(
            X_train_raw=X_train_raw_dt, # Tree 훈련셋 Raw 데이터 (Fit 기준)
            X_test_raw=X_test_raw_reg, # Logit 테스트셋 Raw 데이터 (Transform 대상)
            num_cols=num_cols, 
            cat_cols=cat_cols,
            is_logit=False # Tree 모델 전처리 방식 (Label Encoding/Imputation) 적용
        )
        
        # ----------------------------------------------------------------------
        # B. 분류 (Classification) 평가 로직
        # ----------------------------------------------------------------------
        if st.session_state.task == "logit":
            
            # 1. Logit 모델 예측 (Logit test set 사용)
            prob_reg = reg_model.predict_proba(X_test_reg)[:, 1]
            pred_reg = reg_model.predict(X_test_reg)
            
            # 2. Tree 모델 예측 (Tree test set 사용, 원래 성능)
            prob_dt = dt_model.predict_proba(X_test_dt)[:, 1]
            pred_dt = dt_model.predict(X_test_dt)
            
            # 3. Hybrid 모델 예측 (Logit test set에 Logit, Tree 모두 적용 후 가중치 계산)
            prob_dt_on_reg_test = dt_model.predict_proba(X_test_for_tree_on_logit_set)[:, 1] # 재전처리된 데이터 사용
            prob_hybrid = (prob_reg * w["regression"]) + (prob_dt_on_reg_test * w["decision_tree"])
            pred_hybrid = (prob_hybrid >= 0.5).astype(int)
            
            def get_cls_detailed_metrics(y_true, y_pred, y_prob):
                return {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Recall": recall_score(y_true, y_pred, zero_division=0),
                    "F1-Score": f1_score(y_true, y_pred, zero_division=0),
                    "AUC": auc(*roc_curve(y_true, y_prob)[:2])
                }

            metrics_reg = get_cls_detailed_metrics(y_test_reg, pred_reg, prob_reg)
            metrics_dt = get_cls_detailed_metrics(y_test_dt, pred_dt, prob_dt)
            metrics_hybrid = get_cls_detailed_metrics(y_test_reg, pred_hybrid, prob_hybrid)
            
            # 4. 모델별 성능 비교표 출력
            st.markdown("### 1️⃣ 모델별 주요 성능 지표")
            df_metrics = pd.DataFrame([metrics_reg, metrics_dt, metrics_hybrid], 
                                     index=["Logit Model (Test size: {:.0f}%, OHE/Scaled)".format(st.session_state.models["test_size_reg"]*100), 
                                            "Tree Model (Test size: {:.0f}%, LE/Imputed)".format(st.session_state.models["test_size_dt"]*100), 
                                            "Hybrid Model (Logit Test Set 기준)"])
            st.table(df_metrics.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"))

            # 5. ROC Curve 비교 시각화
            st.markdown("### 2️⃣ ROC Curve 비교")
            fig_roc = go.Figure()
            def add_roc_trace(y_true, y_prob, name, color):
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name, line=dict(color=color, width=2)))

            # ROC Curve는 Logit 테스트셋 기준으로 통일하여 비교
            add_roc_trace(y_test_reg, reg_model.predict_proba(X_test_reg)[:, 1], "Logit", "blue")
            add_roc_trace(y_test_reg, prob_dt_on_reg_test, "Tree (on Logit Test)", "green")
            add_roc_trace(y_test_reg, prob_hybrid, "Hybrid", "red")
            
            fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curves (Logit Test Set 기준)")
            st.plotly_chart(fig_roc, width='stretch')

            # 6. Confusion Matrix (혼동 행렬) 시각화
            st.markdown("### 3️⃣ Confusion Matrix (혼동 행렬)")
            st.caption("각 모델이 정답을 어떻게 맞추고 틀렸는지 시각적으로 확인합니다. (Hybrid는 Logit Test Set 기준)")
            
            cm_col1, cm_col2, cm_col3 = st.columns(3)
            
            def plot_confusion_matrix(y_true, y_pred, title):
                cm = confusion_matrix(y_true, y_pred)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                 labels=dict(x="Predicted", y="Actual", color="Count"),
                                 x=['0 (Neg)', '1 (Pos)'], y=['0 (Neg)', '1 (Pos)'])
                fig.update_layout(title=title, width=300, height=300, margin=dict(l=20, r=20, t=40, b=20))
                return fig

            with cm_col1:
                st.plotly_chart(plot_confusion_matrix(y_test_reg, pred_reg, "Logit Model"), use_container_width=True)
            with cm_col2:
                # Tree 모델은 Logit 테스트셋으로 예측한 결과를 보여줌
                pred_dt_on_reg_test = dt_model.predict(X_test_for_tree_on_logit_set)
                st.plotly_chart(plot_confusion_matrix(y_test_reg, pred_dt_on_reg_test, "Tree Model (on Logit Test)"), use_container_width=True)
            with cm_col3:
                st.plotly_chart(plot_confusion_matrix(y_test_reg, pred_hybrid, "Hybrid Model"), use_container_width=True)

        # ----------------------------------------------------------------------
        # C. 회귀 (Regression) 평가 로직
        # ----------------------------------------------------------------------
        else:
            # 1. Logit 모델 예측
            pred_reg = reg_model.predict(X_test_reg)
            # 2. Tree 모델 예측 (원래 성능)
            pred_dt = dt_model.predict(X_test_dt)
            
            # 3. Hybrid 모델 예측
            pred_dt_on_reg_test = dt_model.predict(X_test_for_tree_on_logit_set) # 재전처리된 데이터 사용
            pred_hybrid = (pred_reg * w["regression"]) + (pred_dt_on_reg_test * w["decision_tree"])
            
            def get_reg_metrics(y_true, y_pred):
                return {
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "R²": r2_score(y_true, y_pred)
                }
            
            m1 = get_reg_metrics(y_test_reg, pred_reg)
            m2 = get_reg_metrics(y_test_dt, pred_dt)
            m3 = get_reg_metrics(y_test_reg, pred_hybrid)
            
            st.markdown("### 1️⃣ 회귀 모델 성능 지표")
            df_reg = pd.DataFrame([m1, m2, m3], 
                                  index=["Logit (Test size: {:.0f}%, OHE/Scaled)".format(st.session_state.models["test_size_reg"]*100), 
                                         "Tree (Test size: {:.0f}%, LE/Imputed)".format(st.session_state.models["test_size_dt"]*100), 
                                         "Hybrid (Logit Test Set 기준)"])
            st.table(df_reg.style.format("{:.4f}"))
            
            st.markdown("### 2️⃣ 예측값 vs 실제값 비교 (Hybrid Model)")
            fig = px.scatter(x=y_test_reg, y=pred_hybrid, title="Hybrid 예측 결과 (Logit Test Set 기준)", labels={'x':'실제값', 'y':'예측값'})
            fig.add_shape(type='line', line=dict(dash='dash', color='red'), x0=y_test_reg.min(), x1=y_test_reg.max(), y0=y_test_reg.min(), y1=y_test_reg.max())
            st.plotly_chart(fig, width='stretch')
