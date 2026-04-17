import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular


# --------------------------
# 页面配置
# --------------------------
st.set_page_config(page_title="糖尿病风险识别", layout="wide")
st.title("糖尿病发病风险识别系统")
st.markdown("---")

# --------------------------
# 加载模型（缓存优化）
# --------------------------
@st.cache_resource
def load_model():
    return joblib.load("XGB.pkl")

model = load_model()

# 固定特征顺序（必须与训练一致）
FEATURE_NAMES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Stroke', 
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 
    'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income', 
    'Age_BMI', 'Cardio_Risk', 'Alcohol_Sex'
]

# --------------------------
# 加载测试集（用于SHAP/LIME）
# --------------------------
@st.cache_data
def load_test_data():
    return pd.read_csv("X_test.csv")

X_test = load_test_data()

# ==============================================
# 【1】用户输入区域（20个特征，全中文+说明）
# ==============================================
st.subheader("请根据自身情况填写以下信息")

input_data = {}

input_data['Sex'] = st.selectbox(
    "性别",
    options=[0, 1],
    format_func=lambda x: "女性" if x == 0 else "男性"
)
input_data['Age'] = st.selectbox(
    "年龄",
    options=[1,2,3,4,5,6,7,8,9,10,11,12,13],
    format_func=lambda x: {
        1:"18岁–24岁",2:"25岁–29岁",3:"30岁–34岁",4:"35岁–39岁",5:"40岁–44岁",6:"45岁–49岁",7:"50岁–54岁",8:"55岁–59岁",9:"60岁–64岁",10:"65岁–69岁",11:"70岁–74岁",12:"75岁–79岁",13:"80岁及以上"
    }[x]
)
input_data['Education'] = st.selectbox(
    "教育水平",
    options=[1,2,3,4,5,6],
    format_func=lambda x: {
        1:"未上过学",2:"小学",3:"初中",4:"高中",5:"大专",6:"本科及以上"
    }[x]
)
input_data['Income'] = st.selectbox(
    "年收入（美元）",
    options=[1,2,3,4,5,6,7,8],
    format_func=lambda x: {
        1:"低于1万",2:"1万～1.5万",3:"1.5万～2万",4:"2万～2.5万",5:"2.5万~3.5万",6:"3.5万~5万",7:"5万~7.5万",8:"高于7.5万"
    }[x]
)
input_data['HighBP'] = st.selectbox(
    "是否有高血压？",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)
input_data['HighChol'] = st.selectbox(
    "是否有高胆固醇？",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)
input_data['CholCheck'] = st.selectbox(
    "是否在5年内检查过胆固醇？",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)
input_data['Stroke'] = st.selectbox(
    "是否有过中风？",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)
input_data['HeartDiseaseorAttack'] = st.selectbox(
    "是否患有冠心病或心肌梗死？",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)

input_data['PhysActivity'] = st.selectbox(
    "近一个月是否有体育运动？",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)
input_data['Fruits'] = st.selectbox(
    "是否每天吃水果？",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)
input_data['HvyAlcoholConsump'] = st.selectbox(
    "是否过度饮酒？（男性>14杯/周，女性>7杯/周）",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)

input_data['DiffWalk'] = st.selectbox(
    "是否存在行走困难？",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)

input_data['Alcohol_Sex'] = st.selectbox(
    "性别与饮酒情况",
    options=[0, 1],
    format_func=lambda x: "女性或男性且不过度饮酒" if x == 0 else "男性且过度饮酒"
)
input_data['Cardio_Risk'] = st.selectbox(
    "心脑血管风险等级",
    options=[0,1,2],
    format_func=lambda x: {
        0:"无高血压且无高胆固醇",
        1:"高血压或高胆固醇",
        2:"两者都有"
    }[x]
)
input_data['GenHlth'] = st.selectbox(
    "总体健康评价",
    options=[1,2,3,4,5],
    format_func=lambda x: {
        1:"好",2:"较好",3:"一般",4:"较差",5:"查"
    }[x]

)

st.markdown("### 🔹 数值信息")
input_data['BMI'] = st.number_input("BMI指数", min_value=0.0, step=0.1)
input_data['MentHlth'] = st.number_input("心理健康天数（近30天内心理不佳天数）", min_value=0, max_value=30, step=1)
input_data['PhysHlth'] = st.number_input("身体健康天数（近30天内身体不适天数）", min_value=0, max_value=30, step=1)
input_data['Age_BMI'] = st.number_input("年龄×BMI指数", min_value=0.0, step=0.1)

st.markdown("---")

# ==============================================
# 【2】预测按钮
# ==============================================
if st.button("开始糖尿病风险预测", type="primary", use_container_width=True):
    with st.spinner("正在预测中，请稍候..."):
        # 构造输入样本
        sample = pd.DataFrame([input_data])[FEATURE_NAMES]
        
        # 模型预测
        prob = model.predict_proba(sample)[0, 1]
        pred_label = "高风险" if prob >= 0.5 else "低风险"


        # ==========================================
        # 展示预测结果
        # ==========================================
        st.subheader("预测结果：")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("风险等级", pred_label)
        with col2:
            st.metric("发病概率", f"{prob:.2%}")


        st.markdown("---")

        # ==========================================
        # 【3】SHAP 力图展示
        # ==========================================
        st.subheader("SHAP特征贡献力图")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        plt.figure(figsize=(12, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values,
            sample,
            matplotlib=True,
            show=False
        )
        st.pyplot(plt, bbox_inches='tight')
        plt.clf()

        # ==========================================
        # 【4】LIME 可解释性分析图
        # ==========================================
        st.subheader("LIME局部可解释性分析")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_test),
            feature_names=FEATURE_NAMES,
            class_names=['低风险', '高风险'],
            mode='classification'
        )
        exp = lime_explainer.explain_instance(
            data_row=sample.values[0],
            predict_fn=model.predict_proba,
            num_features=10
        )
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)
        plt.clf()