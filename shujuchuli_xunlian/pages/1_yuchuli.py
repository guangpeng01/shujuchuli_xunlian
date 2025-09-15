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

# 页面配置
st.set_page_config(
    page_title="数据预处理",   
    layout="wide",    
    initial_sidebar_state="expanded"  
)

@st.cache_data
def load_data(uploaded_file):
    """读取文件"""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None



def leixing_zhuanhuan(df):
    """类型转换的选取"""
    st.subheader('类型选择')    
    
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []    # 存储用户之前选择的列名列表
    if 'column_types' not in st.session_state:
        st.session_state.column_types = {}    # 存储列名到数据类型的映射字典

    with st.form('feature_selection_form'):
        # 第一步：选择要转换的列
        selected_columns = st.multiselect(
            '选择要转换数据类型的列',
            df.columns,
            default=st.session_state.selected_features,
            key='数据列'
        )
        
        # 第二步：为每个选中的列选择类型
        type_mapping = {}
        for col in selected_columns:
            # 获取当前列的默认类型（如果之前已经选择过）
            default_type = st.session_state.column_types.get(col, 'str')
            # 创建列和类型选择器
            selected_type = st.selectbox(
                f'选择 {col} 的数据类型',
                [ 'category','str', 'int', 'float', 'bool', 'datetime'],
                index=['str', 'int', 'float', 'bool', 'category', 'datetime'].index(default_type),
                key=f'类型_{col}'
            )
            type_mapping[col] = selected_type
        
        submitted = st.form_submit_button('确认选择')
        
        if submitted:
            try:
                st.session_state.selected_features = selected_columns
                st.session_state.column_types = type_mapping
                
                st.success(" 数据替换！")
                # 执行类型转换
                for col, dtype in type_mapping.items():
                    try:
                        if dtype == 'datetime':
                            df[col] = pd.to_datetime(df[col])
                        else:
                            df[col] = df[col].astype(dtype)

                        
                    except Exception as e:
                        st.error(f"无法将列 {col} 转换为 {dtype}: {str(e)}")
                        continue
                in_fo(df)
                return df
            except Exception as e:
                st.error(f"数据转换错误: {str(e)}")
                return df
    
    return None


def is_na(df):
    """缺失值填充"""
    st.subheader('缺失值填充及数据的下载')    
    
    # 初始化session_state
    if 'modified_df' not in st.session_state:
        st.session_state.modified_df = df.copy()    # 存储修改后的 DataFrame 副本。
    if 'selected_na_features' not in st.session_state:
        st.session_state.selected_na_features = []    # 记录用户选择的待填充列。
    if 'fill_values' not in st.session_state:
        st.session_state.fill_values = {}    # 存储每列选择的填充值。

    # 使用最新的DataFrame
    current_df = st.session_state.modified_df

    with st.form('na_filling_form'):
        # 第一步：选择要填充的列
        selected_columns = st.multiselect(
            '选择要填充缺失值的列',
            current_df.columns,
            default=st.session_state.selected_na_features,
            key='na_columns'
        )

        # 第二步：为每个选中的列选择填充的值
        fill_mapping = {}
        for col in selected_columns:
            col_type = str(current_df[col].dtype)   # 获取列的类型，并将列类型值转换为索引
            default_fill = st.session_state.fill_values.get(col, None)
            if col_type in ['int64', 'float64']:
                options = [0, int(round(current_df[col].mean())), 
                          int(round(current_df[col].median())), 
                          current_df[col].max(), 
                          current_df[col].min()]
            elif col_type == 'object':
                options = ['unknown', 'missing']
            else:
                options = [None]
                
            fill_value = st.selectbox(
                f'选择 {col} 的填充值',
                options,
                index=options.index(default_fill) if default_fill in options else 0,
                key=f'fill_{col}'
            )
            fill_mapping[col] = fill_value
        
        submitted = st.form_submit_button('确认填充缺失值')
        
        if submitted:
            try:
                # 创建临时副本进行操作
                temp_df = current_df.copy()
                
                # 执行填充
                for col, value in fill_mapping.items():
                    try:
                        temp_df[col] = temp_df[col].fillna(value)
                        st.success(f"列【{col}】填充成功！填充值: {value}")
                    except Exception as e:
                        st.error(f"无法填充列 {col}: {str(e)}")
                        continue
                
                # 更新全局DataFrame
                st.session_state.modified_df = temp_df
                st.session_state.selected_na_features = selected_columns
                st.session_state.fill_values = fill_mapping
                
                # 显示填充后的信息
                st.write("填充后各列缺失值数量:")
                st.dataframe(temp_df.isna().sum()[temp_df.isna().sum() > 0])
                
                in_fo(temp_df)    # 假设是显示DataFrame信息的函数
                st.write('检查无误点击右侧下载数据')
                st.dataframe(temp_df)
                return temp_df
            except Exception as e:
                st.error(f"缺失值填充错误: {str(e)}")
                return current_df
    
    return current_df


def in_fo(data_copy):
    """数据干净度或数据类型"""
    with st.expander("数据干净度或数据类型",expanded=False):
        buffer = io.StringIO()
        data_copy.info(buf=buffer)
        st.code(buffer.getvalue(), language='text')


def descri_be(data_copy):
    """查看处理无效或错误数据"""
    with st.expander("查看处理无效或错误数据",expanded=False):
        st.dataframe(data_copy.describe())


def pingu_qingxi(data_copy):

    in_fo(data_copy)
    descri_be(data_copy)
    converted = leixing_zhuanhuan(data_copy)
    data_copy = converted if converted is not None else data_copy
    
    filled = is_na(data_copy)
    return filled if filled is not None else data_copy

def clear_all():
    """清空会话状态和缓存"""
    # 清空会话状态
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # 清空缓存（如果有使用）
    st.cache_data.clear()
    st.cache_resource.clear()
    

# def clear_all():
#     """清空会话状态和缓存（包括跨页面数据）"""
#     # 1. 获取当前会话的所有键
#     current_keys = list(st.session_state.keys())
    
#     # 2. 保留必要的系统级键（可选）
#     keep_keys = ['_pages', '_session_id']  # Streamlit内部使用的键
    
#     # 3. 删除非系统键
#     for key in current_keys:
#         if key not in keep_keys:
#             del st.session_state[key]
    
#     # 4. 清空所有缓存
#     st.cache_data.clear()
#     st.cache_resource.clear()
    
#     # 5. 强制重置页面（可选）
#     st.rerun()


def main():
    st.title("数据预处理")
    
    # if st.button("重置所有数据"):
    #     clear_all()
        
    uploaded_file = st.file_uploader(
        "上传CSV文件",
        type=["csv"],
        help="请上传包含训练数据的CSV文件"
    )
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            with st.expander("🔍 查看原始数据", expanded=True):
                st.dataframe(df)
            data_copy = df.copy()
            pingu_qingxi(data_copy)

            clear_all()


if __name__ == "__main__":
    main()


