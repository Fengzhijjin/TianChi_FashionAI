# TianChi_FashionAI
### 本项目用于冯芝金简历项目展示

本项目模型在ResNet_v2_50模型的基础上进行了部分迁移与改进，由于计算资源限制的原因，只是单纯将Google训练好ResNet_v2_50模型中卷积部分的变量进行了迁移，并未对卷积部分变量进行训练，只对卷积后的全连接部分了训练，并在模型训练中加入了提前终止、Dropout、数据集增强等模型优化，在2950支队伍中获得了219名，若有足够的计算资源将获得更好的名次。

