# 与 elasticsearch:8.12.2 配套的 IK 中文分词。版本必须一致。
FROM docker.elastic.co/elasticsearch/elasticsearch:8.12.2

USER elasticsearch
RUN elasticsearch-plugin install -b \
    https://release.infinilabs.com/analysis-ik/stable/elasticsearch-analysis-ik-8.12.2.zip
