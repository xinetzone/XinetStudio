更多精彩内容见  [深度学习之旅](https://q735613050.github.io/XinetStudio/)

-----

#### Terminology 中英术语对照表
```py
action, 动作
adversarial learning, 对抗学习
agent, 智能体
attribute space, 属性空间
attribute value, 属性值
attribute, 属性
binary classification, 二分类
classification, 分类
cluster, 簇
clustering, 聚类
confidence, 确信度
contextual bandit problem, 情境式赌博机问题
covariate shift, 协变量转移
credit assignment problem, 信用分配问题
cross-entropy, 交叉熵
data set, 数据集
dimensionality, 维数
distribution, 分布
reinforcement learning, 强化学习
example, 样例
feature vector, 特征向量
feature, 特征
generalization, 泛化
generative adversarial networks, 生成对抗网络
ground-truth, 真相、真实
hypothesis, 假设
independent and identically distributed(i.i.d), 独立同分布
instance, 示例
label space, 标注空间
label, 标注
learing algorithm, 学习算法
learned model, 学得模型
learner, 学习器
learning, 学习
machine translation, 机器翻译
Markov Decision Process, 马尔可夫决策过程
model, 模型
multi-armed bandit problem, 多臂赌博机问题
multi-class classification, 多分类
negative class, 反类
offline learning, 离线学习
positive class, 正类
prediction, 预测
principal component analysis, 主成分分析
regression, 回归
reinforcement learning, 强化学习
representation learning, 表征学习
sample space, 样本空间
sample, 样本
sepecilization, 特化
sequence learning, 序列学习
subspace estimation, 子空间估计
supervised learning, 监督学习
testing sample, 测试样本
testing, 测试
time step, 时间步长
training data, 训练数据
training sample, 训练样本
training set, 训练集
training, 训练
unsupervised learning, 无监督学习
```

-----

动手去实现、去调参、去跑实验才会真正成为专家（或者合格的炼丹师）：[深度学习·炼丹入门](https://zhuanlan.zhihu.com/p/23781756)

# 安装

首先安装 Anaconda，之后直接运行下面命令，安装 GPU 版本

```sh
pip install -U mxnet-cu90
```

# [使用 NDArray 来处理数据](http://zh.gluon.ai/chapter_crashcourse/ndarray.html)
