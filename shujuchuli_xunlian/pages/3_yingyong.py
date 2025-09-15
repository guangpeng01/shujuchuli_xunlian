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
        
        # # 添加提交按钮（居中显示）
        # col1, col2, col3 = st.columns([1,2,1])
        # with col2:
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
    """清空会话状态和缓存"""
    # 清空会话状态
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # 清空缓存（如果有使用）
    st.cache_data.clear()
    st.cache_resource.clear()


def main():
    st.title("模型应用")   

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
                with st.expander("🔍 查看原始数据", expanded=False):
                    st.dataframe(df.head())
                
                df_processed = data_preprocessing(df_new_copy)
                if df_processed is not None:
                    df_processed_x = feature_selection(df_processed)
                    
                    if df_processed_x is not None:
                        ying_yong(pickle_model, df_processed_x, df)
                        clear_all()
                        
        except Exception as e:
            st.error(f"加载模型或数据出错: {str(e)}")


if __name__ == "__main__":
    main()








# class DataProcessor:
#     def __init__(self, df):
#         """初始化数据处理类"""
#         self.raw_df = df.copy()
#         self._init_session_state()
        
#     def _init_session_state(self):
#         """初始化会话状态"""
#         if 'processed_data' not in st.session_state:
#             st.session_state.processed_data = self.raw_df.copy()
#         if 'selected_features' not in st.session_state:
#             st.session_state.selected_features = self._get_default_features()
    
#     def _get_default_features(self):
#         """获取默认特征（前3个数值型列）"""
#         numeric_cols = self.raw_df.select_dtypes(include=np.number).columns.tolist()
#         return numeric_cols[:min(3, len(numeric_cols))] if numeric_cols else self.raw_df.columns.tolist()[:1]
    
#     def show_correlation_matrix(self, df):
#         """显示相关性矩阵"""
#         st.subheader('特征相关性矩阵')
#         try:
#             corr_matrix = df.corr()  
#             fig, ax = plt.subplots(figsize=(10,6))   
#             sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
#             st.pyplot(fig)    
#         except Exception as e:
#             st.warning(f"无法计算相关性矩阵: {str(e)}")
    
#     def preprocess(self):
#         """数据预处理界面"""
#         st.subheader('数据预处理')
        
#         # 检查是否需要重置数据
#         if not st.session_state.processed_data.equals(self.raw_df):
#             st.session_state.processed_data = self.raw_df.copy()
        
#         with st.form('preprocessing_form'):   
#             col1, col2 = st.columns(2)    
            
#             with col1:
#                 columns_to_drop = st.multiselect(
#                     '选择要删除的列',
#                     st.session_state.processed_data.columns,
#                     help="选择不需要的特征列"
#                 )
                
#             with col2:
#                 categorical_cols = st.session_state.processed_data.select_dtypes(
#                     include=['object', 'category']).columns.tolist()  
#                 columns_to_encode = st.multiselect(
#                     '选择要进行独热编码的列',
#                     categorical_cols,   
#                     help="选择分类变量进行独热编码"
#                 )
            
#             submitted = st.form_submit_button('应用预处理') 
        
#         if submitted:
#             try:
#                 # 执行删除列
#                 if columns_to_drop:    
#                     st.session_state.processed_data = st.session_state.processed_data.drop(
#                         columns=columns_to_drop)
                
#                 # 执行独热编码
#                 if columns_to_encode:    
#                     st.session_state.processed_data = pd.get_dummies(
#                         st.session_state.processed_data, 
#                         columns=columns_to_encode, 
#                         dtype=int
#                     )
                
#                 # 重置特征选择
#                 if 'selected_features' in st.session_state:
#                     del st.session_state.selected_features

#                 st.success("预处理完成！")
#                 self.show_correlation_matrix(st.session_state.processed_data)
                
#             except Exception as e:
#                 st.error(f"预处理出错: {str(e)}")
        
#         return st.session_state.processed_data
    
#     def select_features(self):
#         """特征选择界面"""
#         st.subheader('特征选择')    
        
#         # 如果特征选择不存在或与当前数据不匹配，则重置
#         current_cols = set(st.session_state.processed_data.columns)
#         if ('selected_features' not in st.session_state) or \
#            (not set(st.session_state.selected_features).issubset(current_cols)):
#             st.session_state.selected_features = self._get_default_features()
        
#         with st.form('feature_selection_form'):
#             # 显示当前特征状态
#             st.markdown(f"当前可选特征数: **{len(current_cols)}**")
            
#             # 特征选择器
#             features = st.multiselect(
#                 '选择特征变量',
#                 st.session_state.processed_data.columns,
#                 default=st.session_state.selected_features
#             )
            
#             # 操作按钮
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.form_submit_button('确认选择'):
#                     if not features:
#                         st.error("❌ 请至少选择一个特征变量！")
#                         return None
#                     st.session_state.selected_features = features
#                     st.success(f"✅ 已选择 {len(features)} 个特征")
#                     return st.session_state.processed_data[features]
#             with col2:
#                 if st.button('重置选择'):
#                     st.session_state.selected_features = self._get_default_features()
#                     st.rerun()
        
#         return st.session_state.processed_data[st.session_state.selected_features]


# 使用示例
# def main():
#     st.title("数据联动处理演示")
    
#     uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         processor = DataProcessor(df)
        
#         with st.expander("原始数据", expanded=False):
#             st.dataframe(df.head())
        
#         # 数据预处理
#         processed_data = processor.preprocess()
        
#         # 特征选择
#         if processed_data is not None:
#             selected_data = processor.select_features()
#             if selected_data is not None:
#                 st.success("最终选择的数据:")
#                 st.dataframe(selected_data)
#                 clear_all()