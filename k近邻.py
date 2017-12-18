import numpy as np
import operator
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
def classify0(inX,dataSet,lables,k):
    #获取总样本个数
    dataSetSize=dataSet.shape[0]
    #求每个样本与分类样本的欧式距离
    diffMat =np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #按照距离从小到大排序，这里的argsort是得到排完序的元素的索引，比如如果第二个元素最小则信息xx[0]=1
    sortedDistIndicies = distances.argsort()
    classCount={}
    #取出与分类样本距离最近的前k个样本
    for i in range(k):
        voteIlabel = lables[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
#这个k个样本中，哪个标签最多，该分类样本被分为哪一类
    return sortedClassCount[0][0]
def file2matrix(filename):
    #将文本数据转换为样本集
    fr=open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #初始化样本向量,行数为样本数，列数为三
    returnMat = zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        #截取所有回车字符
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector
#归一化特征
def autoNormal(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(shape(dataSet))
    m = dataSet.shape[0]
    normdataSet = dataSet - np.tile(minVals,(m,1))
    return normdataSet,ranges,minVals
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datinglabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    num
def main():
    group,labels=createDataSet()
    print(classify0([0,0],group,labels,3))
main()
