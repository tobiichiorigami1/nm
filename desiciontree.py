from math import log
import operator
#求香农熵，香农熵等于各个分类在总样本中的所占比例对2求对数的相反数之和（因为比例小于1，所以对数为负数要取相反数）
#有点像求各个类的最大似然之和对2取对数再求相反数
def calcShannonEnt(dataSet):
#得出样本总个数
    numEntries = len(dataSet)
    labelCounts={}
    #得出各个类的样本数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
            labelCounts[currentLabel]+=1
        else:
            labelCounts[currentLabel]+=1
            shannonEnt = 0.0
    for key in labelCounts:
        #求各个分类样本个数占总样本数的比例（最大似然）
        prob = float(labelCounts[key])/numEntries
        print(prob)
        #取对数相反数相加得到香农熵
        shannonEnt-=prob*log(prob,2)
    return shannonEnt
def createDataSet():
    dataSet=[[1,1,'maybe'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels
def splitDataSet(dataSet,axis,value):
    retDataSet =[]
#遍历样本集中的样本
    for featVec in dataSet:
        #将样本符合划分特征值的每个样本除了需划分的特征的特征值以外的所有值存取得到新数据集
        if(featVec[axis]==value):
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
        retDataSet.append(reducedFeatVec)
    return retDataSet
def chooseBestFeatureToSplit(dataSet):
    #获取特征个数,每个样本总长度减标签一
    numFeatures = len(dataSet[0]-1)
#求该样本集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
#初始化最大熵增益，最好决策特征
    bestInfoGain = 0.0
    bestFeature = -1
    #循环按每一个特征将数据集划分
    #首先获取每一个特征可能的取值
    for i in range(numFeatures):
#这里还可以弄一层循环
        featList=[]
        for example in dataSet:
            featList.append(example[i])
#将特征的可能取值的每种值只取一次
        uniqueVals = set(featList)
#求取每个特征的香农熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)
#求增益熵
        infoGain = baseEntropy-newEntropy
#如果当前求得增益熵大于原存储的增益熵，更新最好的增益熵值（即求出所有增益熵中最大的那个）
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
#记录下最大熵增益所在特征的索引
    return bestFeature
def majorityCnt(classList):
    classCount=()
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]    
#创建决策树
def createtree(dataSet,labels):
#将样本的所有标签取出来
    classList = [example[-1] for example in dataSet]
#如果所有样本的标签都是同一个，则把该样本集作为根结点
    if classList.count(classList[0]) == len(classList):
        return classList[0]
#
    if len(dataSet[0])==1:
        return majorityCnt{classList}
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    mytree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        sublabels = labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet\(dataSet,bestFeat,value),subLabels)
    return myTree
def main():
    data,label=createDataSet()
    shan=calcShannonEnt(data)
    print(shan)
    df=splitDataSet(data,1,1)
    print(df)
main()
