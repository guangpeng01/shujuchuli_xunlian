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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


# 页面配置
st.set_page_config(
    page_title="向量化处理",   
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
    """特征选择界面"""
    st.subheader('特征选择')    
    
    
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = None
    
    
    with st.form('feature_selection_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            features = st.multiselect(
                '选择特征变量',
                df.columns,
                default=st.session_state.selected_features,    
                key='feature_select'
            )
            
        with col2:
            target = st.selectbox(
                '选择目标变量',
                df.columns,
                index=df.columns.get_loc(st.session_state.selected_target) 
                if st.session_state.selected_target in df.columns else 0,
                key='target_select'
            )
        
        submitted = st.form_submit_button('确认选择')

        if submitted:
            if not features:
                st.error("❌ 请至少选择一个特征变量！")
                return None, None
            
            if target in features:
                st.error("❌ 目标变量不能同时作为特征变量！")
                return None, None
            
            try:
                st.session_state.selected_features = features
                st.session_state.selected_target = target
                
                st.success("✅ 特征选择有效！")
                return df[features].values, df[target].values
            except Exception as e:
                st.error(f"数据选择错误: {str(e)}")
                return None, None

    return None, None


# def auto_save_model(model):
#     """自动保存模型到当前脚本所在目录"""
#     try:
#         # 获取当前脚本所在目录
#         script_dir = os.path.dirname(os.path.abspath(__file__))
        
#         # 生成带时间戳的文件名
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"auto_model_{timestamp}.pkl"
#         save_path = os.path.join(script_dir, filename)
        
#         # 保存模型
#         with open(save_path, "wb") as file:
#             pickle.dump(model, file)
        
#         st.success(f"✅ 模型已自动保存到: {save_path}")
#         return True
#     except Exception as e:
#         st.error(f"❌ 自动保存失败: {str(e)}")
#         return False

def auto_save_model(model):
    """自动保存模型并提供明确的用户指引"""
    try:
        # 1. 创建专用保存目录
        save_dir = os.path.join(os.path.expanduser("~"), "auto_saved_models")
        os.makedirs(save_dir, exist_ok=True)
        
        # 2. 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_{timestamp}.pkl"
        save_path = os.path.join(save_dir, filename)
        
        # 3. 保存模型文件
        with open(save_path, "wb") as file:
            pickle.dump(model, file)
        
        # 4. 显示完整的用户指引
        st.success("✅ 模型保存成功！")
        
        # 显示文件信息卡片
        with st.expander("📁 模型文件信息", expanded=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("保存位置", value="用户目录/auto_saved_models")
            with col2:
                st.code(f"完整路径: {save_path}")
            
            st.write("如何找到这个文件：")
            st.markdown("""
            - **Windows**: 打开文件资源管理器 → 输入路径 `%USERPROFILE%\auto_saved_models`
            - **Mac/Linux**: 打开终端 → 运行 `open ~/auto_saved_models` 或 `cd ~/auto_saved_models`
            """)
        
        # 5. 直接提供下载按钮
        with open(save_path, "rb") as file:
            st.download_button(
                label="⬇️ 立即下载模型文件",
                data=file,
                file_name=filename,
                mime="application/octet-stream"
            )
        
        return True
    except Exception as e:
        st.error(f"❌ 保存失败: {str(e)}")
        st.error("请尝试以下解决方案：")
        st.markdown("""
        1. 检查磁盘空间是否充足
        2. 确保您有写入权限（特别是Linux/Mac系统）
        3. 尝试手动指定保存位置：
        """)
        
        # 添加手动选择路径的备用方案
        custom_path = st.text_input("或输入自定义保存路径（如：C:/models/）")
        if st.button("手动保存"):
            if custom_path:
                try:
                    os.makedirs(custom_path, exist_ok=True)
                    custom_save = os.path.join(custom_path, filename)
                    with open(custom_save, "wb") as f:
                        pickle.dump(model, f)
                    st.success(f"手动保存成功！路径: {custom_save}")
                except Exception as manual_error:
                    st.error(f"手动保存失败: {str(manual_error)}")
        
        return False

def train_random_forest(X, y):
    """训练模型"""
    st.subheader('模型训练')
     # ​惰性初始化  仅在第一次运行时初始化，后续保留训练结果
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    try:
        st.session_state.training_results = None    # ​强制重置 每次调用函数时清空历史结果
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
      
        if len(np.unique(y_train)) < 6:
            # 训练模型
            model = RandomForestClassifier()
            model = model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)  # 获取预测结果
            results = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),   # 评价得分
                'f1': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'features': X.shape[1]
            }
    
        else:
           
            model = GradientBoostingRegressor()  
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)  # 获取预测结果
            # mse = mean_squared_error(y_test, y_pred)
            # rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            results = {
                'model': model,
                # 'mean': mse,
                'r2': r2,
                'features': X.shape[1]
            }

        # 保存结果
        st.session_state.training_results = results
        st.session_state.model = model
        st.success('训练完成！')
        if 'accuracy' in results:  # 分类任务
            st.metric("准确率", f"{results['accuracy']:.4f}")
            st.metric("F1分数", f"{results['f1']:.4f}")
            st.write("混淆矩阵:", results['confusion_matrix'])
        else:  # 回归任务
            # st.metric("均方误差", f"{results['mean']:.4f}")
            st.metric("R²分数", f"{results['r2']:.4f}")

        auto_save_model(model)  # 调用自动保存函数
       
    except Exception as e:
        st.error(f"训练异常: {str(e)}")
    
    # 自动显示训练结果

    return None


def clear_all():
    """清空会话状态和缓存"""
    # 清空会话状态
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # 清空缓存（如果有使用）
    st.cache_data.clear()
    st.cache_resource.clear()
    

def main():
    st.title("数据向量化")   

    uploaded_file = st.file_uploader(
        "上传CSV文件",
        type=["csv"],
        help="请上传包含训练数据的CSV文件"
    )
    # clear_all()
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            with st.expander("🔍 查看原始数据", expanded=True):
                st.dataframe(df)
            
            df_processed = data_preprocessing(df)

            if df_processed is not None:
                # 特征选择
                X, y = feature_selection(df_processed)
                
                if X is not None and y is not None:
                    train_random_forest(X, y)

                    clear_all()
    # return None                


if __name__ == "__main__":

    main()
