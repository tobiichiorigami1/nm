import numpy as np
#创建样本
def loadDataSet():
    postingList=[['my','dog','has','flea','problem'\
                  ,'help','please'],\
                 ['maybe','not','take','him'\
                  ,'to','dog','park','stupid'],\
                 ['my','dalmation','is','so','cute',\
                  'I','love','him'],['stop','posting',\
                    'stupid','worthless','garbage'],\
                 ['mr','licks','ate','my','steak','how',\
                  'to','stop','him'],['quit','buying',\
                'worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec
#从已知样本中归纳出单词表并去重
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        print(document)
        vocabSet=vocabSet | set(document)
    return list(vocabSet)
#将每个样本转化为词向量
def setOfWord2Vec(vocabList,inputSet):
    returnvec=np.zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnvec[vocabList.index(word)]+=1
        else: print("词汇表找不到该词")
    return returnvec
#训练朴素贝叶斯函数
def trainNB0(trainMatrix,trainCategory):
    #计算总样本个数
    numTrainDocs=len(trainMatrix)
    #计算每个样本词向量的维度，维度一样
    numWords = len(trainMatrix[0])
    #计算所有样本中，为侮辱样本的比例，标签为1,0，然后和刚好为侮辱样本的个数。
    #这个概率，在贝叶斯中称为似然。
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #0初始化各个概率
    p0Num =np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        #如果该样本为侮辱样本
        if trainCategory[i]==1:
            #把该样本的单词每个个数加1
            p1Num+=trainMatrix[i]
            #求出该样本的所有词数并加起来
            p1Denom+=sum(trainMatrix[i])
#如果样本为好样本
        else:
            #同上
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
#求取先验概率,这里用到python的广播特性，比如已知该样本为侮辱样本，求某个词出现的概率
            #其实就等于某个词在侮辱样本中出现的总次数除以总词数。
    p1Vect=np.log(p1Num/p1Denom)
    p0Vect=np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
#定义贝叶斯分类器
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #输入样本为侮辱的概率等于先验概率乘以最大似然概率除以一个常数，常数比例因子与最终结果判定无关因为两边都要乘该常数
    #这里用词向量去乘以概率向量，因为可能词向量为0，不存在该词，则不必要乘以该概率.
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
#输入样本为非侮辱的概率，同理
    p0=sum(vec2Classify*p0Vec)+np.log(1-pClass1)
#比较哪个概率大，哪个概率大把它归于哪一类
    if p1 > p0:
        return 1
    else:
        return 0
def main():
    data,vec=loadDataSet()
    print(data)
    list1=createVocabList(data)
    print(list1)
    cc=[]
    for i in data:
        cc.append(setOfWord2Vec(list1,i))
    print(cc)
    p1,p2,p3=trainNB0(cc,vec)
    print(p1,p2,p3)
    x=[]
    for i in cc:
        x.append(classifyNB(i,p1,p2,p3))
    print(x)
      
main()
