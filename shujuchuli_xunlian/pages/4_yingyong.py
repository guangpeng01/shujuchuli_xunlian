import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import io

# from xianglianghua import data_preprocessing,clear_all,load_data

# 页面配置
st.set_page_config(
    page_title="数据应用练习",   
    layout="wide",    
    initial_sidebar_state="expanded"  
)

# 缓存数据加载
@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None

def in_fo(data_copy):
    """数据干净度或数据类型"""
    with st.expander("数据干净度或数据类型",expanded=False):
        buffer = io.StringIO()
        data_copy.info(buf=buffer)
        st.code(buffer.getvalue(), language='text')


def data_preprocessing(df):
    """数据预处理函数"""
    st.subheader('数据预处理')

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = df.copy()
    
    with st.form('preprocessing_form'):   
        # st.session_state.training_results = None    # ​强制重置 每次调用函数时清空历史结果
        col1, col2 = st.columns(2)    
        
        with col1:
            columns_to_drop = st.multiselect(
                '选择要删除的列',
                st.session_state.processed_data.columns,
                
                help="选择不需要的特征列"
            )
            
        with col2:
            categorical_cols = st.session_state.processed_data.select_dtypes(
                include=['object', 'category','int']).columns.tolist()  
            columns_to_encode = st.multiselect(
                '选择要进行独热编码的列',
                categorical_cols,   
                
                help="选择分类变量进行独热编码"
            )
        
        submitted = st.form_submit_button('应用预处理') 
    
    if submitted:
        try:
            
            if columns_to_drop:    
                st.session_state.processed_data = st.session_state.processed_data.drop(
                    columns=columns_to_drop)
            
          
            if columns_to_encode:    
                st.session_state.processed_data = pd.get_dummies(
                    st.session_state.processed_data, 
                    columns=columns_to_encode, 
                    dtype=int
                )
            
            if 'selected_features' in st.session_state:
                del st.session_state.selected_features


            st.success("预处理完成！")
            show_correlation_matrix(st.session_state.processed_data)

            
        except Exception as e:
            st.error(f"预处理出错: {str(e)}")
    
    return st.session_state.processed_data    


def show_correlation_matrix(df):
    """显示相关性矩阵"""
    st.subheader('特征相关性矩阵')
    try:
        corr_matrix = df.corr()  
        fig, ax = plt.subplots(figsize=(10,6))   
        sns.heatmap(
            corr_matrix,    
            annot=True,    
            fmt=".2f",     
            cmap='coolwarm',  
            center=0,    
            ax=ax      
        )
        # plt.title("特征相关性热力图", pad=20)  
        st.pyplot(fig)    
    except Exception as e:
        st.warning(f"无法计算相关性矩阵: {str(e)}")


def feature_selection(df):
    """特征选择界面（单列布局）"""
    st.subheader('特征选择')    

    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []

    # 使用表单
    with st.form('feature_selection_form'):
        # 单列布局（不需要使用columns）
        features = st.multiselect(
            '选择特征变量',
            df.columns,
            default=st.session_state.selected_features,    
            key='feature_select'
        )
        
        submitted = st.form_submit_button('确认选择')

    if submitted:
        if not features:
            st.error("❌ 请至少选择一个特征变量！")
            return None
        
        try:
            st.session_state.selected_features = features
            st.success("✅ 特征选择有效！")
            return df[features]
        except Exception as e:
            st.error(f"数据选择错误: {str(e)}")
            return None
    
    return None


def ying_yong(pickle_model, df_new_copy, df_new):
    try:
        # 检查模型是否有feature_names_in_属性
        if hasattr(pickle_model, 'feature_names_in_'):
            required_features = pickle_model.feature_names_in_
            # 确保数据包含所有需要的特征
            missing_features = set(required_features) - set(df_new_copy.columns)
            if missing_features:
                for feature in missing_features:
                    df_new_copy[feature] = 0  # 填充缺失特征为0
            df_new_copy = df_new_copy[required_features]
        
        y_pred = pickle_model.predict(df_new_copy.values)
        df_new['predict'] = y_pred
        st.dataframe(df_new)
    except Exception as e:
        st.error(f"预测出错: {str(e)}")


def clear_all():
    """清空会话状态和缓存（包括跨页面数据）"""
    # 1. 获取当前会话的所有键
    current_keys = list(st.session_state.keys())
    
    # 2. 保留必要的系统级键（可选）
    keep_keys = ['_pages', '_session_id']  # Streamlit内部使用的键
    
    # 3. 删除非系统键
    for key in current_keys:
        if key not in keep_keys:
            del st.session_state[key]
    
    # 4. 清空所有缓存
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # 5. 强制重置页面（可选）
    st.rerun()

def main():
    st.title("模型应用")   

    if st.button("重置所有数据"):
        clear_all()

    uploaded_file0 = st.file_uploader('选择本地模型', type=['pkl'])
    uploaded_file = st.file_uploader(
        "上传CSV文件",
        type=["csv"],
        help="请上传包含训练数据的CSV文件"
    )

    if (uploaded_file is not None) and (uploaded_file0 is not None):
        try:
            pickle_model = pickle.load(uploaded_file0)
            
            df = load_data(uploaded_file)

            if df is not None:
                df_new_copy = df.copy()
                with st.expander("🔍 查看原始数据", expanded=True):
                    st.dataframe(df)
                    in_fo(df)
                
                df_processed = data_preprocessing(df_new_copy)
                if df_processed is not None:
                    df_processed_x = feature_selection(df_processed)
                    
                    if df_processed_x is not None:
                        ying_yong(pickle_model, df_processed_x, df)
                        # clear_all()
                        
        except Exception as e:
            st.error(f"加载模型或数据出错: {str(e)}")


if __name__ == "__main__":
    main()
