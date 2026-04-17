import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular

# --------------------------
# 页面配置 + 自定义CSS 【核心修改1：控制整体居中、容器宽度】
# --------------------------
st.set_page_config(page_title="糖尿病风险识别", layout="centered")

# 自定义CSS：限制内容最大宽度、所有模块居中、调整间距、图表统一尺寸
st.markdown("""
<style>
/* 整体内容容器最大宽度，和参考图一致收窄居中 */
.block-container {
    max-width: 700px !important;
    padding-top: 2rem;
    padding-bottom: 2rem;
    margin: 0 auto;
}
/* 按钮居中+适配大小 */
.stButton > button {
    width: 100% !important;
    border-radius: 8px;
}
/* 标题居中 */
h1, h2, h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("糖尿病风险识别")

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
# 【1】用户输入区域 保持居中窄容器
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
        1:"小学以下",2:"小学",3:"初中",4:"高中",5:"大专",6:"本科及以上"
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
        1:"好",2:"较好",3:"一般",4:"较差",5:"差"
    }[x]

)
input_data['BMI'] = st.number_input("BMI指数", min_value=0.0, step=0.1)
input_data['MentHlth'] = st.number_input("近30天内心理状况不佳天数", min_value=0, max_value=30, step=1)
input_data['PhysHlth'] = st.number_input("近30天内身体状况不适天数", min_value=0, max_value=30, step=1)
input_data['Age_BMI'] = st.number_input("年龄×BMI指数（其中年龄取值为年龄栏所选行数）", min_value=0.0, step=0.1)


# ==============================================
# 【2】预测按钮
# ==============================================
if st.button("一键预测", type="primary"):
    with st.spinner("正在预测中，请稍候..."):
        # 构造输入样本
        sample = pd.DataFrame([input_data])[FEATURE_NAMES]
        
        # 模型预测
        prob = model.predict_proba(sample)[0, 1]
        prediction = 1 if prob >= 0.5 else 0
        # ==========================================
        # 展示预测结果 【修改2：排版居中、文字描述和参考图对齐】
        # ==========================================
        st.markdown("---")
        st.subheader("预测结果")
        
        # 文字说明 + 概率展示，和参考图文案风格统一
        st.write(f"**预测类别：{prediction} (0=不患有糖尿病, 1=患有糖尿病)**")
        st.write(f"**预测概率：不患有糖尿病:{1-prob:.3f}，高风险:{prob:.3f}**")
        if prob < 0.5:
            st.info(f"根据模型判断，您目前糖尿病患病风险较低，无病概率为 **{(1-prob)*100:.1f}%**，请保持健康作息、定期体检。")
        else:
            st.warning(f"根据模型判断，您目前糖尿病患病风险较高，患病概率为 **{prob*100:.1f}%**，建议及时就医检查、调整生活习惯。")

        st.markdown("---")

        # ==========================================
        # 【3】SHAP 力图展示 【修改3：固定图表尺寸，窄长条横向，和参考图一模一样比例】
        # ==========================================
        st.subheader("SHAP Force Plot 特征解释")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        
        # 强制设置 宽12 高3 的横条图，和示例图尺寸比例完全匹配
        plt.figure(figsize=(12, 3), dpi=100)
        shap.force_plot(
            explainer.expected_value,
            shap_values,
            sample,
            matplotlib=True,
            show=False,
            text_rotation=0
        )
        # 图表居中展示
        st.pyplot(plt, bbox_inches='tight', use_container_width=True)
        plt.clf()
        plt.close("all") # 彻底释放画布，避免多图重叠

        st.markdown("---")

        # ==========================================
        # 【4】LIME 可解释性分析图 【修改4：统一图表大小、居中对齐】
        # ==========================================
        st.subheader("LIME 局部特征解释")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_test),
            feature_names=FEATURE_NAMES,
            class_names=['低风险', '高风险'],
            mode='classification',
            random_state=42
        )
        exp = lime_explainer.explain_instance(
            data_row=sample.values[0],
            predict_fn=model.predict_proba,
            num_features=10
        )
        # 固定LIME图尺寸，匹配整体排版
        fig = exp.as_pyplot_figure(label=1)
        fig.set_size_inches(10, 6)
        st.pyplot(fig, bbox_inches='tight', use_container_width=True)
        plt.clf()
        plt.close("all")