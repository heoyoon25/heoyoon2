import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# SciKit-Learn 및 기타 모듈 가져오기 (오류 방지를 위해 try-except 문 제거 후, 설치 가정)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# --- 1. 모델 학습 및 평가 함수 정의 ---

@st.cache_data(show_spinner="데이터를 로드하고 전처리 중...")
def load_and_preprocess(uploaded_file):
    """파일을 로드하고 데이터 전처리를 수행합니다."""
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None
    
    # 컬럼명 정제 및 핵심 변수 정의 (사용자 데이터에 맞게 조정 필요)
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True).str.lower()
    df.rename(columns={'loan_amount': 'loan_amount', 'riskscore': 'risk_score',
                       'debttodincome_ratio': 'dti', 'loantitle': 'loan_title',
                       'employmentlength': 'emp_length', 'policycode': 'policy_code',
                       'state': 'state'}, inplace=True)
    
    # 핵심 변수 선택
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

# @st.cache_data를 사용하여 모델 학습 및 평가 결과를 캐시
@st.cache_data(show_spinner="모델 학습 중...")
def run_model(X_train, Y_train, X_test, Y_test, model_name):
    """선택된 모델을 학습하고 평가합니다."""
    if model_name == '로지스틱 회귀분석':
        model = LogisticRegression(random_state=42, max_iter=2000, solver='liblinear')
    elif model_name == '의사결정나무':
        model = DecisionTreeClassifier(random_state=42, max_depth=5)
    
    # NaN 최종 확인 및 0으로 대치
    X_train_final = X_train.fillna(0)
    X_test_final = X_test.fillna(0)

    model.fit(X_train_final, Y_train)
    Y_pred = model.predict(X_test_final)
    Y_pred_proba = model.predict_proba(X_test_final)[:, 1]

    # 성능 지표 계산
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    recall = recall
