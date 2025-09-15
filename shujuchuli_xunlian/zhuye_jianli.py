import pandas as pd
import numpy as np
import streamlit as st

st.markdown("""## 简历""")

st.markdown("""#### 姓名：常广鹏""")

st.markdown("""#### 电话：17662513419(微信同号)""")


st.markdown("""
### 应用介绍

这是一个基于 **Streamlit** 构建的数据分析平台，支持以下功能：

- 数据预处理：上传预处理文件 ——> 格式转换 ——> 空缺值填补
            
- 数据向量化：上传需要向量化的文件 ——> 选择删除不需要的列和需要向量化的列 ——> 对向量化后的热力图的绘制以方便查看之间的相关系数
            
- 模型的训练：根据热力图的展示选择相关的数据进行训练 ——> 查看训练结果
            
- 模型的应用：上传下载后的模型和所需要预测的数据 ——> 选择删除不需要的列和需要向量化的列 ——> 根据热力图的展示选择相关的数据进行预测
            
- 数据缓存会话存在一些问题：可以刷新页面
            
- 欢迎大佬点评
            

""")


# st.markdown("""### 代码及实例数据的获取地址""")

st.markdown("[代码及实例数据的获取地址](https://github.com/guangpeng01/shujuchuli_xunlian)")

st.markdown("##### 看的懂一些基础的python、pandas和numpy代码")


st.markdown("""##### 求职岗位:数据岗 """)
