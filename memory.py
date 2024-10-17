# 显存分配
# 情况1.如果你是在 Python 脚本或 Jupyter Notebook 中运行的，需要通过 os.environ 来设置环境变量。
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"