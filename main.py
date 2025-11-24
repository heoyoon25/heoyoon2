import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# SciKit-Learn 및 기타 모듈 가져오기 (설치 필요: scikit-learn, matplotlib, seaborn)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
except ImportError:
    st.error("❌ 필요한 라이브러리가 설치되지 않았습니다. 다음 명령으로 설치하세요: pip install scikit-learn matplotlib seaborn")
    st.stop()


# --- 전역 변수 및 함수 설정 ---
# @st.cache_data를 사용하여 파일 업로드 시에만 데이터 로드/전처리 실행
@st.cache_data(show_spinner="데이터를 로드하고 전처리 중...")
def load_and_preprocess(uploaded_file):
    """파일을 로드하고 데이터 전처리를 수행합니다."""
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None, None

    # 1. 컬럼명 정제 (일관성을 위해 띄어쓰기 및 특수문자 제거)
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True).str.lower()
    df.rename(columns={'loan_amount': 'loan_amount', 'riskscore': 'risk_score',
                       'debttodincome_ratio': 'dti', 'loantitle': 'loan_title',
                       'employmentlength': 'emp_length', 'policycode': 'policy_code',
                       'state': 'state'}, inplace=True)
    
    # 2. 핵심 변수 정의 및 선택 (사용자 데이터에 맞게 조정)
    required_cols = ['loan_amount', 'risk_score', 'dti', 'state', 'loan_title', 'emp_length', 'loan_status']
    available_cols = [col for col in required_cols if col in df.columns]
    df_model = df[available_cols].copy()

    # 3. 데이터 타입 및 결측치 처리 (Imputation)
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
    
    return df_model, df.columns.tolist()

# @st.cache_data를 사용하여 전처리된 데이터를 기반으로 모델 학습 및 평가 결과를 캐시
@st.cache_data(show_spinner="모델 학습 중...")
def run_model(X_train, Y_train, X_test, Y_test, model_name):
    """선택된 모델을 학습하고 평가합니다."""
    if model_name == '로지스틱 회귀분석':
        model = LogisticRegression(random_state=42, max_iter=2000, solver='liblinear')
    elif model_name == '의사결정나무':
        model = DecisionTreeClassifier(random_state=42, max_depth=5)
    
    # NaN 최종 확인 및 0으로 대치 (모델 오류 방지)
    X_train_final = X_train.fillna(0)
    X_test_final = X_test.fillna(0)

    model.fit(X_train_final, Y_train)
    Y_pred = model.predict(X_test_final)
    Y_pred_proba = model.predict_proba(X_test_final)[:, 1]

    # 성능 지표 계산
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)
    roc_auc = roc_auc_score(Y_test, Y_pred_proba)

    return model, accuracy, precision, recall, f1, roc_auc, Y_pred_proba, Y_test


# --- Streamlit 앱 메인 함수 ---
def main():
    st.set_page_config(layout="wide", page_title="랜딩클럽 ML 분석 앱")
    st.title("💸 랜딩클럽 대출 승인 예측 분석 시스템")

    # 세션 상태 초기화
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = None
    if 'df_model' not in st.session_state:
        st.session_state.df_model = None

    # --- 사이드바 메뉴 설정 ---
    st.sidebar.title("메뉴")
    menu = st.sidebar.radio("원하는 분석 단계를 선택하세요.", 
                            ("파일 업로드", "데이터 시각화", "데이터 전처리", 
                             "데이터 나누기", "모델 선택", "예측 및 성능 평가"))
    
    # --- 1. 파일 업로드 ---
    if menu == "파일 업로드":
        st.header("1. 파일 업로드 📂")
        uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])

        if uploaded_file is not None:
            df_model, original_cols = load_and_preprocess(uploaded_file)
            st.session_state.df_model = df_model
            
            if df_model is not None:
                st.success("파일 업로드 및 기본 전처리 완료!")
                st.dataframe(df_model.head())
                st.markdown(f"**총 {len(df_model)} 행**의 데이터가 로드되었습니다.")
            
    # 나머지 메뉴는 데이터가 로드된 후에만 접근 가능
    elif st.session_state.df_model is not None:
        
        df_model = st.session_state.df_model
        
        # --- 2. 데이터 시각화 ---
        if menu == "데이터 시각화":
            st.header("2. 데이터 시각화 📊")
            
            # 컬럼 타입별 분리
            numeric_cols = df_model.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_label = st.selectbox("X-Label (범주형 변수)", categorical_cols)
            with col2:
                y_label = st.selectbox("Y-Label (수치형 변수)", numeric_cols)
            with col3:
                chart_type = st.selectbox("그래프 종류", [
                    "막대 그래프 (Bar Plot)", "상자 수염 그림 (Box Plot)", 
                    "히스토그램 (Histogram)", "산점도 (Scatter Plot)", 
                    "바이올린 그림 (Violin Plot)", "히트맵 (Heatmap)"
                ])

            st.markdown("---")
            fig, ax = plt.subplots(figsize=(10, 6))

            if chart_type == "막대 그래프 (Bar Plot)":
                sns.countplot(data=df_model, x=x_label, ax=ax)
                ax.set_title(f'{x_label}의 빈도 분석')
            elif chart_type == "상자 수염 그림 (Box Plot)":
                sns.boxplot(data=df_model, x=x_label, y=y_label, ax=ax)
                ax.set_title(f'{x_label}별 {y_label} 분포')
            elif chart_type == "히스토그램 (Histogram)":
                sns.histplot(df_model[y_label], kde=True, ax=ax)
                ax.set_title(f'{y_label}의 빈도 분포')
            elif chart_type == "산점도 (Scatter Plot)":
                # 산점도는 두 수치형 변수가 필요하므로, 다른 수치형 변수를 y_label로 설정하거나 기본값을 사용
                # 여기서는 x_label에 범주형, y_label에 수치형을 유지하고, 범주형을 X축으로 분리하여 산점도 대신 Stripplot 사용
                sns.stripplot(data=df_model, x=x_label, y=y_label, ax=ax, jitter=True)
                ax.set_title(f'{x_label}별 {y_label} 관측치')
            elif chart_type == "바이올린 그림 (Violin Plot)":
                sns.violinplot(data=df_model, x=x_label, y=y_label, ax=ax)
                ax.set_title(f'{x_label}별 {y_label} 밀도 분포')
            elif chart_type == "히트맵 (Heatmap)":
                # 히트맵은 상관관계를 보기 위해 모든 수치형 변수를 사용합니다.
                corr_matrix = df_model[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title('변수 간 상관관계 히트맵')
            
            # X축 레이블이 너무 길 경우 회전
            if len(df_model[x_label].unique()) > 10:
                plt.xticks(rotation=45, ha='right')

            st.pyplot(fig) # 

        # --- 3. 데이터 전처리 ---
        elif menu == "데이터 전처리":
            st.header("3. 데이터 전처리 과정 확인 👀")
            st.markdown("로지스틱 회귀 모델 학습을 위한 데이터 변환 과정입니다.")
            
            st.subheader("1단계: 결측치 중앙값/범주 대치")
            st.markdown("- 수치형 변수(`loan_amount`, `risk_score`, `dti`): 중앙값(Median)으로 결측치 처리")
            st.markdown("- 범주형 변수(`state`, `loan_title`, `emp_length`): 'Missing' 범주로 결측치 처리")
            st.dataframe(df_model.head())
            
            st.subheader("2단계: 범주형 변수 원-핫 인코딩 (One-Hot Encoding)")
            st.markdown("모델 학습을 위해 범주형 변수를 숫자형 더미 변수로 변환합니다. (분할 직전 단계)")
            
            # 원-핫 인코딩 시뮬레이션
            temp_df = df_model.drop(columns=['loan_status']).copy()
            final_categorical_cols = temp_df.select_dtypes(include=['object', 'category']).columns.tolist()
            temp_df = pd.get_dummies(temp_df, columns=final_categorical_cols, drop_first=True)
            st.dataframe(temp_df.head())


        # --- 4. 데이터 나누기 ---
        elif menu == "데이터 나누기":
            st.header("4. 데이터 나누기 (Train/Test Split) ✂️")
            
            split_ratio = st.slider("훈련 데이터 비율을 선택하세요.", 0.5, 0.9, 0.8, 0.05)
            st.info(f"선택 비율: **훈련(Train): {split_ratio * 100:.0f}%**, **테스트(Test): {(1 - split_ratio) * 100:.0f}%**")
            
            st.session_state.split_ratio = split_ratio


        # --- 5. 모델 선택 ---
        elif menu == "모델 선택":
            st.header("5. 모델 선택 🧠")
            
            model_choice = st.selectbox("사용할 분류 모델을 선택하세요.", ['로지스틱 회귀분석', '의사결정나무'])
            st.session_state.model_choice = model_choice
            
            if 'split_ratio' not in st.session_state:
                st.warning("데이터 분할 비율을 먼저 설정해주세요 (4. 데이터 나누기 메뉴).")
            else:
                st.success(f"현재 선택된 모델: **{st.session_state.model_choice}**")


        # --- 6. 예측 및 성능 평가 ---
        elif menu == "예측 및 성능 평가":
            st.header("6. 예측 및 성능 평가 📈")

            if 'split_ratio' not in st.session_state or 'model_choice' not in st.session_state:
                st.warning("모델 학습을 위해 '데이터 나누기'와 '모델 선택'을 먼저 완료해야 합니다.")
                return

            # 데이터 분할 비율 및 모델 선택 정보 가져오기
            ratio = st.session_state.split_ratio
            model_name = st.session_state.model_choice

            # ----------------------------------------------------
            # 모델 학습을 위한 최종 데이터 준비 (전처리 재실행)
            # ----------------------------------------------------
            temp_df = st.session_state.df_model.copy()
            X = temp_df.drop('loan_status', axis=1)
            Y = temp_df['loan_status']

            final_categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            X = pd.get_dummies(X, columns=final_categorical_cols, drop_first=True)
            
            # 훈련/테스트 분할
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=(1 - ratio), random_state=42, stratify=Y
            )
            
            # 수치형 변수 표준화 (다시 적용)
            numeric_cols = [col for col in ['loan_amount', 'risk_score', 'dti'] if col in X_train.columns]
            scaler = StandardScaler()
            X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])


            # 모델 실행 및 평가
            try:
                model, acc, prec, rec, f1, roc_auc, y_proba, Y_test_final = run_model(X_train, Y_train, X_test, Y_test, model_name)
                
                st.success(f"**{model_name}** 모델 학습 및 평가 완료!")

                st.subheader("주요 성능 지표")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Precision", f"{prec:.4f}")
                col3.metric("Recall", f"{rec:.4f}")
                col4.metric("F1 Score", f"{f1:.4f}")
                col5.metric("ROC AUC", f"{roc_auc:.4f}")

                st.markdown("---")
                
                # ROC 곡선 시각화
                st.subheader("ROC 곡선 (Receiver Operating Characteristic)")
                fpr, tpr, thresholds = roc_curve(Y_test_final, y_proba)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
                ax.set_xlabel('False Positive Rate (FPR)')
                ax.set_ylabel('True Positive Rate (TPR)')
                ax.set_title(f'{model_name} ROC Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)

                # 혼동 행렬
                st.subheader("혼동 행렬 (Confusion Matrix)")
                conf_matrix = confusion_matrix(Y_test_final, model.predict(X_test_final.fillna(0)))
                st.text(conf_matrix)


            except Exception as e:
                st.error(f"모델 학습 중 오류 발생: {e}")
                st.error("데이터를 다시 확인하거나 다른 모델을 선택해 보세요.")
    else:
        st.info("시작하려면 '파일 업로드' 메뉴에서 CSV 파일을 업로드해주세요.")

if __name__ == '__main__':
    main()
