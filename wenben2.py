'''文本相似度比较（13步）'''
import jieba
from gensim import corpora, models, similarities
from collections import defaultdict

#1、读取文档
doc1=open("A:/fiction/ljm.txt").read()
doc2=open("A:/fiction/gcd.txt").read()
#2、对要计算的多篇文档进行分词
data1 = jieba.cut(doc1)
data2 = jieba.cut(doc2)
#3、对文档进行整理成指定格式，方便后续进行计算
data11 = ""
for item in data1:
    data11 += item+" "
data21 = ""
for item in data2:
    data21 += item+" "
documents = [data11, data21]
texts = [[word for word in document.split()] for document in documents]
#4、计算出词语的频率
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
#5【可选】、对频率低的词语进行过滤
texts = [[word for word in text if frequency[token]>1] for text in texts]
#6、通过语料库建立词典
dictionary = corpora.Dictionary(texts)
dictionary.save("A:/fiction/dict345.txt")
#7、加载要对比的文档
doc3=open("A:/fiction/gcd.txt").read()
data3 = jieba.cut(doc3)
data31 = ""
for item in data3:
    data31 += item+" "
new_doc = data31
#8、将要对比的文档通过doc2bow转化为稀疏向量
new_vec=dictionary.doc2bow(new_doc.split())
#9、对稀疏向量进行进一步处理，得到新语料库
corpus=[dictionary.doc2bow(text) for text in texts]
#10、将新语料库通过tfidfmodel进行处理，得到tfidf
corpora.MmCorpus.serialize("A:/fiction/d3.mm",corpus)
tfidf=models.TfidfModel(corpus)
#11、通过token2id得到特征数
featureNum=len(dictionary.token2id.keys())
#12、计算稀疏矩阵相似度，从而建立索引
index=similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=featureNum)
#13、比较稀疏矩阵，得到最终相似度结果
sim=index[tfidf[new_vec]]
print(sim)







