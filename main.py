import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- 1. 모델 학습 및 평가 함수 정의 ---

@st.cache_data
def load_and_preprocess_data(file_path):
    """데이터를 로드하고 로지스틱 회귀에 맞게 전처리합니다."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except FileNotFoundError:
        st.error(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None, None, None, None, None, None
    
    # 이전에 정의된 핵심 변수 목록 (실제 컬럼명에 따라 수정 필요)
    SELECTED_FEATURES = [
        'Loan_Amount', 'Loan Title', 'Risk_Score', 'Debt-To-Income Ratio', 
        'State', 'Employment Length', 'Policy Code', 'Loan_Status'
    ]
    available_cols = [col for col in SELECTED_FEATURES if col in df.columns]
    df_model = df[available_cols].copy()

    # 데이터 타입 변환 및 결측치 대치 (Imputation)
    numeric_cols = ['Loan_Amount', 'Risk_Score', 'Debt-To-Income Ratio', 'Policy Code']
    categorical_cols_impute = ['Loan Title', 'State', 'Employment Length']

    for col in numeric_cols:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

    for col in [c for c in ['Loan_Amount', 'Risk_Score', 'Debt-To-Income Ratio'] if c in df_model.columns]:
        median_value = df_model[col].median()
        df_model[col] = df_model[col].fillna(median_value if not np.isnan(median_value) else 0)

    for col in categorical_cols_impute:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna('Missing')

    # 범주형 변수 처리 (원-핫 인코딩)
    categorical_cols = [col for col in categorical_cols_impute if col in df_model.columns]
    if 'Policy Code' in df_model.columns and df_model['Policy Code'].nunique() < 50:
        categorical_cols.append('Policy Code')

    df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

    # 특징(X)과 종속 변수(Y) 분리 및 분할
    if 'Loan_Status' not in df_model.columns:
        st.error("❌ 'Loan_Status' 종속 변수가 데이터에 없습니다.")
        return None, None, None, None, None, None
        
    X = df_model.drop('Loan_Status', axis=1)
    Y = df_model['Loan_Status']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # 수치형 변수 표준화
    final_numeric_features = [col for col in numeric_cols if col in X_train.columns]
    if 'Policy Code' in final_numeric_features and 'Policy Code' in categorical_cols:
        final_numeric_features.remove('Policy Code') 

    scaler = StandardScaler()
    if final_numeric_features:
        X_train.loc[:, final_numeric_features] = scaler.fit_transform(X_train[final_numeric_features])
        X_test.loc[:, final_numeric_features] = scaler.transform(X_test[final_numeric_features])

    # 최종 결측치 처리 (0으로 강제 대치하여 모델 오류 방지)
    X_train_final = X_train.fillna(0)
    X_test_final = X_test.fillna(0)
    
    return X_train_final, Y_train, X_test_final, Y_test, X.columns, final_numeric_features


@st.cache_resource
def train_logistic_regression(X_train, Y_train):
    """로지스틱 회귀 모델을 학습시킵니다."""
    model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    model.fit(X_train, Y_train)
    return model

# --- 2. Streamlit UI 구성 ---

st.title("💳 랜딩클럽 대출 승인 예측 모델 (로지스틱 회귀)")
st.write("`combined_loan_data.csv` 파일을 사용하여 로지스틱 회귀 모델을 학습시키고 성능을 분석합니다.")

# 파일 경로 설정 (사용자 환경에 맞게 수정 필요)
data_file_path = 'combined_loan_data.csv'

# 데이터 로드 및 전처리 실행
X_train, Y_train, X_test, Y_test, feature_names, numeric_features = load_and_preprocess_data(data_file_path)

if X_train is not None and X_train.shape[0] > 0:
    st.sidebar.header("모델 학습 및 분석")
    st.sidebar.markdown(f"**훈련 데이터 크기:** {X_train.shape[0]} 행")
    st.sidebar.markdown(f"**테스트 데이터 크기:** {X_test.shape[0]} 행")
    
    # 모델 학습
    model = train_logistic_regression(X_train, Y_train)

    st.subheader("모델 학습 결과 및 성능 평가")

    # 예측 및 성능 평가
    Y_pred = model.predict(X_test)
    Y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred_proba)

    # 주요 성능 지표 요약
    col1, col2, col3 = st.columns(3)
    col1.metric("정확도 (Accuracy)", f"{accuracy:.4f}")
    col2.metric("ROC AUC", f"{roc_auc:.4f}")
    col3.metric("특징 개수", f"{X_train.shape[1]}")

    
    # 탭을 사용하여 정보 분리
    tab1, tab2, tab3 = st.tabs(["상세 보고서", "특징 중요도", "ROC 곡선"])

    with tab1:
        st.subheader("상세 분류 보고서")
        report = classification_report(Y_test, Y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        st.subheader("혼동 행렬")
        st.text(conf_matrix)

    with tab2:
        st.subheader("모델 계수 (특징 중요도)")
        
        # 모델 계수 (Coefficients) 추출 및 데이터프레임으로 변환
        coefficients = model.coef_[0]
        feature_importance = pd.Series(coefficients, index=X_train.columns).sort_values(ascending=False)
        
        st.bar_chart(feature_importance.head(20))
        st.write("로지스틱 회귀 계수는 특징이 승인 확률(Y=1)에 미치는 영향을 나타냅니다.")
        
    with tab3:
        st.subheader("ROC 곡선 시각화")
        
        # ROC 곡선 그리기
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)

else:
    st.warning("데이터 로드 또는 전처리 중 문제가 발생했습니다. `combined_loan_data.csv` 파일을 확인하거나, 데이터에 유효한 샘플이 없습니다.")
