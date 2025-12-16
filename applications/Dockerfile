FROM openwhisk/python3action:latest

RUN apk add --update py-pip

# 升级 pip
RUN pip install --upgrade pip

# 安装系统依赖
RUN apk add --no-cache \
    libjpeg-turbo-dev \
    zlib-dev \
    libpng-dev

# 安装 Python 依赖包（移除了 torch 相关）
RUN pip install --no-cache-dir \
    requests>=2.25.1 \
    igraph>=0.9.0 \
    Pillow>=8.3.1 \
    markdown>=3.3.4 \
    textblob>=0.17.1 \
    numpy>=1.21.0 \
    squiggle

# 下载 TextBlob 语料库
RUN python -m textblob.download_corpora

# 清理缓存
RUN rm -rf /var/cache/apk/* && \
    rm -rf /root/.cache/pip/*