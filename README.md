终端输入如下命令，创建项目的虚拟环境，并指定Python版本：
conda create -n product-classify python=3.12
激活该虚拟环境：
conda activate product-classify
本项目依赖以下软件和库：
pytorch：深度学习框架，用于训练和推理
transformers：Hugging Face 提供的库，用于加载和微调 BERT 等预训练模型。
datasets：用于高效加载和处理大规模数据集。
scikit-learn：用于模型评估。
tensorboard：用于可视化训练过程中的损失、准确率等指标。
tqdm：用于显示训练进度条，方便监控训练过程。
jupyter：用于实验和数据分析。
FastAPI：用于构建和部署API接口。 
Uvicorn：FastAPI的服务器，用于高性能地运行FastAPI应用。
