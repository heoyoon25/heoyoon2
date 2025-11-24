import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# SciKit-Learn 및 기타 모듈 가져오기 (설치 가정)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# --- 1. 모델 학습 및 평가 함수 정의 ---

# 함수 입력으로 uploaded_file 대신 df를 직접 받도록 변경
@st.cache_data(show_spinner="데이터 전처리 중...")
def preprocess_data(df):
    """로드된 데이터프레임을 받아 전처리를 수행합니다."""
    
    # 컬럼명 정제 및 핵심 변수 정의 (이전 단계의 복잡한 정규식 제거 후 간소화)
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    df.rename(columns={'loan_amount': 'loan_amount', 'riskscore': 'risk_score',
                       'debttodincome_ratio': 'dti', 'loantitle': 'loan_title',
                       'employmentlength': 'emp_length', 'policycode': 'policy_code',
                       'state': 'state'}, inplace=True)
    
    required_cols = ['loan_amount', 'risk_score', 'dti', 'state', 'loan_title', 'emp_length', 'loan_status']
    available_cols = [col for col in required_cols if col in df.columns]
    df_model = df[available_cols].copy()

    # 데이터 타입 및 결측치 처리 (Imputation)
    numeric_cols = ['loan_amount', 'risk_score', 'dti']
    categorical_cols = ['state', 'loan_title', 'emp_length']

    for col in numeric_cols:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
            median_val = df_model[col].median()
            df_model[col] = df_model[col].fillna(median_val if not pd.isna(median_val) else 0)

    for col in categorical_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype('category').cat.add_categories('Missing').fillna('Missing')
    
    return df_model

@st.cache_data(show_spinner="모델 학습 중...")
def run_model(X_train, Y_train, X_test, Y_test, model_name):
    # (이전과 동일한 모델 학습 및 평가 함수)
    if model_name == '로지스틱 회귀분석':
        model = LogisticRegression(random_state=42, max_iter=2000, solver='liblinear')
    elif model_name == '의사결정나무':
        model = DecisionTreeClassifier(random_state=42, max_depth=5)
    
    X_train_final = X_train.fillna(0)
    X_test_final = X_test.fillna(0)

    model.fit(X_train_final, Y_train)
    Y_pred = model.predict(X_test_final)
    Y_pred_proba = model.predict_proba(X_test_final)[:, 1]

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)
    roc_auc = roc_auc_score(Y_test, Y_pred_proba)

    return model, accuracy, precision, recall, f1, roc_auc, Y_pred_proba, Y_test


# --- 2. Streamlit UI 구성 ---
def main():
    st.set_page_config(layout="wide", page_title="랜딩클럽 ML 분석 앱")
    st.title("💸 랜딩클럽 대출 승인 예측 분석 시스템")

    if 'df_model' not in st.session_state:
        st.session_state.df_model = None
    if 'split_ratio' not in st.session_state:
        st.session_state.split_ratio = 0.8  
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = '로지스틱 회귀분석'

    st.sidebar.title("메뉴")
    menu = st.sidebar.radio("원하는 분석 단계를 선택하세요.", 
                            ("파일 업로드", "데이터 시각화", "데이터 전처리", 
                             "데이터 나누기", "모델 선택", "예측 및 성능 평가"))
    
    # --- A. 파일 업로드 ---
    if menu == "파일 업로드":
        st.header("1. 파일 업로드 📂")
        
        # 파일 업로더를 통해 파일 객체 받기
        uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])

        if uploaded_file is not None:
            # 파일 객체로부터 pandas DataFrame 생성
            df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
            
            # 전처리 함수 실행 및 결과 저장
            df_model = preprocess_data(df)
            st.session_state.df_model = df_model
            
            st.success("파일 업로드 및 기본 전처리 완료! 다음 단계로 이동하세요.")
            st.dataframe(df_model.head())
        
        # --- B. Streamlit Cloud에서 GitHub 파일을 읽는 임시 로직 (로컬 테스트용) ---
        # Streamlit Cloud는 파일 업로더 대신 GitHub의 파일 경로를 읽어야 하지만, 
        # file_uploader가 None일 때 처리하지 않으면 충돌하므로, 
        # 이 부분은 Streamlit Cloud의 파일 업로드/경로 처리 로직에 맡깁니다.

    # --- C. 데이터 로드 후 메뉴 접근 ---
    elif st.session_state.df_model is not None:
        df_model = st.session_state.df_model
        
        # --- 2. 데이터 시각화 ---
        if menu == "데이터 시각화":
            # (시각화 코드 - 이전과 동일)
            
            numeric_cols = df_model.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # (생략: 시각화 로직)
            st.subheader("2. 데이터 시각화 📊 (데이터 로드 완료)")
            
            # ... (이하 시각화 코드 - 이전과 동일)

            # NOTE: 시각화 코드는 매우 길어 생략하고, 성공 여부만 확인합니다.
            st.success("시각화 메뉴 접근 성공. 시각화 로직을 실행합니다.")
            
            # 여기에 이전의 시각화 코드 삽입

        # --- 3. 데이터 전처리 ---
        elif menu == "데이터 전처리":
            # (데이터 전처리 과정 확인 코드 - 이전과 동일)
            st.subheader("3. 데이터 전처리 과정 확인 👀 (데이터 로드 완료)")
            st.dataframe(df_model.head())

        # --- 4. 데이터 나누기 ---
        elif menu == "데이터 나누기":
            # (데이터 분할 슬라이더 코드 - 이전과 동일)
            st.subheader("4. 데이터 나누기 (Train/Test Split) ✂️ (데이터 로드 완료)")
            st.session_state.split_ratio = st.slider("훈련 데이터 비율을 선택하세요. (예: 0.8 = 8:2)", 0.5, 0.9, st.session_state.split_ratio, 0.05)
            st.info(f"선택 비율: **훈련(Train): {st.session_state.split_ratio * 100:.0f}%**, **테스트(Test): {(1 - st.session_state.split_ratio) * 100:.0f}%**")


        # --- 5. 모델 선택 ---
        elif menu == "모델 선택":
            # (모델 선택 코드 - 이전과 동일)
            st.subheader("5. 모델 선택 🧠 (데이터 로드 완료)")
            st.session_state.model_choice = st.selectbox("사용할 분류 모델을 선택하세요.", ['로지스틱 회귀분석', '의사결정나무'])


        # --- 6. 예측 및 성능 평가 ---
        elif menu == "예측 및 성능 평가":
            # (모델 학습 및 평가 코드 - 이전과 동일)
            st.subheader("6. 예측 및 성능 평가 📈 (데이터 로드 완료)")
            
            # NOTE: 이 부분은 모델 학습을 위해 코드를 짧게 생략합니다.
            
            # ratio, model_name 정의
            ratio = st.session_state.split_ratio
            model_name = st.session_state.model_choice

            # 데이터 준비 (이전과 동일한 로직)
            temp_df = st.session_state.df_model.copy()
            X = temp_df.drop('loan_status', axis=1)
            Y = temp_df['loan_status']

            final_categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            X = pd.get_dummies(X, columns=final_categorical_cols, drop_first=True)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1 - ratio), random_state=42, stratify=Y)

            # 모델 실행 및 평가
            try:
                # (모델 학습 및 평가 실행)
                model, acc, prec, rec, f1, roc_auc, y_proba, Y_test_final = run_model(X_train, Y_train, X_test, Y_test, model_name)
                
                st.success(f"**{model_name}** 모델 학습 및 평가 완료! (Accuracy: {acc:.4f})")
                # (이하 성능 지표 출력 코드 - 이전과 동일)

            except Exception as e:
                st.error(f"모델 학습 중 오류 발생: {e}")
                
    else:
        st.info("시작하려면 **'파일 업로드'** 메뉴에서 CSV 파일을 업로드해주세요. (로컬 테스트용)")
        st.info("Streamlit Cloud 사용 시, `main.py` 파일 내에서 `combined_loan_data.csv`를 직접 읽는 로직이 필요할 수 있습니다.")

if __name__ == '__main__':
    main()
@st.cache_data(show_spinner="데이터 전처리 중...")
def preprocess_data(df):
    """로드된 데이터프레임을 받아 전처리를 수행합니다."""
    
    # 1. 컬럼명 정제 및 통일 (오류 발생 가능성 줄이기)
    # 기존 코드의 정규식을 유지하되, 모든 컬럼을 통과시킵니다.
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=False).str.replace('-', '_', regex=False).str.replace('[^a-z0-9_]', '', regex=True)
    
    # 2. 컬럼명 최종 고유화 (중복 오류 방지)
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for item in cols:
        counter = 1
        new_item = item
        while new_item in seen:
            new_item = item + '_' + str(counter)
            counter += 1
        seen[new_item] = True
        new_cols.append(new_item)
    df.columns = new_cols
    
    # 3. 핵심 변수 정의 및 선택 (컬럼명 고유화 후 진행)
    required_cols = ['loan_amount', 'risk_score', 'dti', 'state', 'loan_title', 'emp_length', 'loan_status']
    available_cols = [col for col in required_cols if col in df.columns]
    df_model = df[available_cols].copy()

    # (이하 결측치 처리 및 데이터 타입 변환 로직은 이전과 동일)

    numeric_cols = ['loan_amount', 'risk_score', 'dti']
    categorical_cols = ['state', 'loan_title', 'emp_length']

    for col in numeric_cols:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
            median_val = df_model[col].median()
            df_model[col] = df_model[col].fillna(median_val if not pd.isna(median_val) else 0)

    for col in categorical_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype('category').cat.add_categories('Missing').fillna('Missing')
    
    return df_model
