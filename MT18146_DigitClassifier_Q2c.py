import utils.mnist_reader
import matplotlib.pyplot as plt
import math
import numpy
import copy

def binarize(dataset):
    binarized_dataset=[]
    for i in range(0,len(dataset)):
        data = dataset[i]
        binary_data=[]
        for j in range(0,len(data)):
            if data[j]<80:
                binary_data.append(0)
            else:
                binary_data.append(3)
        binarized_dataset.append(binary_data)

    return binarized_dataset

def mixingdataset(total_fold,total_fold_label):
    one_set=[]
    eight_set=[]
    for j in range(0,len(total_fold_label)):
        if  total_fold_label[j]==3:
            one_set.append(total_fold[j])
        elif total_fold_label[j]==8:
            eight_set.append(total_fold[j])
    print("oneset: ",len(one_set),"eightset :",len(eight_set))
    diff = abs(len(one_set)-len(eight_set))
    if len(one_set)>len(eight_set):
        one_set=one_set[:len(eight_set)]
    else:
        eight_set=eight_set[:len(one_set)]
    print(len(one_set),len(eight_set))
    final_dataset=[]
    final_label =[]
    for i in range(0,5):
        final_dataset+=one_set[i*int(len(one_set)/5):(i+1)*int(len(one_set)/5)]
        temp=[3]*int(len(one_set)/5)
        final_label+=temp
        temp=[]
        final_dataset+=eight_set[i*int(len(one_set)/5):(i+1)*int(len(eight_set)/5)]
        temp=[8]*int(len(eight_set)/5)
        final_label+=temp
        temp=[]
    return final_dataset,final_label

def calculate5folding(i,total_fold,total_fold_label,part):
    if i==0:
        test_fold_dataset=total_fold[:part]
        test_fold_label=total_fold_label[:part]
        train_fold_dataset=total_fold[part:]
        train_fold_label=total_fold_label[part:]
    elif i==1:
        test_fold_dataset=total_fold[part:2*part]
        test_fold_label=total_fold_label[part:2*part]
        train_fold_dataset=total_fold[:part]+total_fold[2*part:]
        train_fold_label=total_fold_label[:part]+total_fold_label[2*part:]
    elif i==2:
        test_fold_dataset=total_fold[2*part:3*part]
        test_fold_label=total_fold_label[2*part:3*part]
        train_fold_dataset=total_fold[:2*part]+total_fold[3*part:]
        train_fold_label=total_fold_label[:2*part]+total_fold_label[3*part:]
    elif i==3:
        test_fold_dataset=total_fold[3*part:4*part]
        test_fold_label=total_fold_label[3*part:4*part]
        train_fold_dataset=total_fold[:3*part]+total_fold[4*part:]
        train_fold_label=total_fold_label[:3*part]+total_fold_label[4*part:]
    elif i==4:
        test_fold_dataset=total_fold[4*part:5*part]
        test_fold_label=total_fold_label[4*part:5*part]
        train_fold_dataset=total_fold[:4*part]
        train_fold_label=total_fold_label[:4*part]
    return test_fold_dataset,test_fold_label,train_fold_dataset,train_fold_label

def calculate_likelihood(mean,feature,sigma):
    likelihood=0
    for i in range(0,len(feature)):
        if sigma[i]==0:
            sigma[i]=0.1
        temp = 1/(pow(2*3.14*sigma[i],0.5))*math.exp(-1*(pow(feature[i]-mean[i],2)/(2*sigma[i])))
        if temp!=0:
            likelihood+=math.log10(temp)
    return likelihood

def calculate_posterior_probability(likelihood,prior):
    posterior_probability = likelihood+math.log10(prior)
    return posterior_probability

def DET_curve_points(posterior_trouser_dataset,binarized_test_dataset,label_test_dataset,actual_test_trouser,actual_test_pullover):
    posterior = copy.copy(posterior_trouser_dataset)
    posterior_trouser_dataset.sort()
    x_points=[]
    y_points=[]
    for threshold in range(0,len(posterior_trouser_dataset)):
        fp=0
        fn=0
        for i in range(0,len(binarized_test_dataset)):
            #feature = binarized_test_dataset[i]
            #likelihood_trouser = calculate_likelihood(mean_trouser,feature,variance_trouser)
            #posterior_probability_trouser = calculate_posterior_probability(likelihood_trouser,actual_test_trouser/(actual_test_pullover+actual_test_trouser))
            if posterior[i] >= posterior_trouser_dataset[threshold]:
                if label_test_dataset[i]!=3:
                    fp+=1
            else:
                if label_test_dataset[i]==3:
                    fn+=1
        y_points.append(fp/actual_test_pullover)
        x_points.append(fn/actual_test_trouser)
        #print(threshold,fp/actual_test_pullover,tp/actual_test_trouser)

    return x_points,y_points

def ROC_curve_points(posterior_trouser_dataset,binarized_test_dataset,label_test_dataset,actual_test_trouser,actual_test_pullover):
    posterior = copy.copy(posterior_trouser_dataset)
    posterior_trouser_dataset.sort()
    x_points=[]
    y_points=[]
    for threshold in range(0,len(posterior_trouser_dataset)):
        fp=0
        tp=0
        for i in range(0,len(binarized_test_dataset)):
            #feature = binarized_test_dataset[i]
            #likelihood_trouser = calculate_likelihood(mean_trouser,feature,variance_trouser)
            #posterior_probability_trouser = calculate_posterior_probability(likelihood_trouser,actual_test_trouser/(actual_test_pullover+actual_test_trouser))
            if posterior[i] >= posterior_trouser_dataset[threshold]:
                if label_test_dataset[i]==3:
                    tp+=1
                else:
                    fp+=1
        y_points.append(fp/actual_test_pullover)
        x_points.append(tp/actual_test_trouser)
        #print(threshold,fp/actual_test_pullover,tp/actual_test_trouser)

    return x_points,y_points

X_train, y_train = utils.mnist_reader.load_mnist('data/mnist', kind='train')
X_test, y_test = utils.mnist_reader.load_mnist('data/mnist', kind='t10k')

train_dataset=[]
train_label_dataset=[]
test_dataset=[]
test_label_dataset=[]
for i in range(0,len(X_train)):
    if y_train[i]==3 or y_train[i]==8:
        train_dataset.append(X_train[i])
        #if y_train[i]==1:
        train_label_dataset.append(y_train[i])
for i in range(0,len(X_test)):
    if y_test[i]==3 or y_test[i]==8:
        test_dataset.append(X_test[i])
        test_label_dataset.append(y_test[i])

confusion_matrix=[[0,0],[0,0]]
total_fold_size=len(train_dataset)+len(test_dataset)
total_fold=train_dataset+test_dataset
total_fold_label=train_label_dataset+test_label_dataset
print(len(total_fold),len(total_fold_label))
div=total_fold_size%5
part=int(total_fold_size/5)
pr_list=[]
for i in range(0,5):
    if i!=4:
    #    print("hi")
        continue
    #print("hee")
    final_dataset,final_label=mixingdataset(total_fold,total_fold_label)
    print(len(final_dataset),len(final_label))
    test_fold_dataset,test_fold_label,train_fold_dataset,train_fold_label=calculate5folding(i,final_dataset,final_label,part)
    print(len(train_fold_dataset),len(train_fold_label))
    train_1_dataset=[]
    train_8_dataset=[]
    #print(train_fold_label,train_fold_dataset)
    for j in range(0,len(train_fold_label)):
        if train_fold_label[j]==3:
            train_1_dataset.append(train_fold_dataset[j])
        if train_fold_label[j]==8:
            train_8_dataset.append(train_fold_dataset[j])
    actual_1=0
    actual_8=0
    for j in range(0,len(test_label_dataset)):
        if test_label_dataset[j]==3:
            actual_1+=1
        elif test_label_dataset[j]==8:
            actual_8+=1
    #print("train 1: ",train_1_dataset)
    #print("train 8 :",train_8_dataset)
    #print("test : ",test_fold_label)
    binarized_1_dataset=binarize(train_1_dataset)
    binarized_8_dataset=binarize(train_8_dataset)
    binarized_test_dataset=binarize(test_fold_dataset)
    #print("b 1 : ",binarized_1_dataset)
    #print("b 8 : ",binarized_8_dataset)
    #print("n t : ",binarized_test_dataset)
    mean_1 = numpy.mean(binarized_1_dataset,axis=0)
    mean_8 = numpy.mean(binarized_8_dataset,axis=0)
    variance_1 = numpy.var(binarized_1_dataset,axis=0)
    variance_8 = numpy.var(binarized_8_dataset,axis=0)
    #print(variance_1)
    true=0
    predicted_1=0
    predicted_8=0
    posterior_3_dataset=[]
    binarized_test_dataset=binarize(test_dataset)
    for i in range(0,len(binarized_test_dataset)):
        feature = binarized_test_dataset[i]
        likelihood_1 = calculate_likelihood(mean_1,feature,variance_1)
        posterior_probability_1 = calculate_posterior_probability(likelihood_1,actual_1/(actual_1+actual_8))
        likelihood_8 = calculate_likelihood(mean_8,feature,variance_8)
        posterior_probability_8 = calculate_posterior_probability(likelihood_8,actual_8/(actual_1+actual_8))
        posterior_3_dataset.append(posterior_probability_1)
        label=0
        if posterior_probability_1 > posterior_probability_8:
            label=3
        else:
            label=8
        if label==test_label_dataset[i]:
            true+=1
            if label==3:
                confusion_matrix[0][0]+=1
            if label==8:
                confusion_matrix[1][1]+=1
        elif label==3 and test_label_dataset[i]==8:
            confusion_matrix[0][1]+=1
        elif label==8 and test_label_dataset[i]==3:
            confusion_matrix[1][0]+=1
        if label==3:
          predicted_1+=1
        else:
          predicted_8+=1
    print("accuracy : ",true/len(test_label_dataset))
    pr_list.append(true/len(test_label_dataset))

#x_points,y_points=ROC_curve_points(posterior_3_dataset,binarized_test_dataset,train_label_dataset,actual_1,actual_8)
#plt.plot(y_points,x_points)
#plt.xlabel("False postive rate")
#plt.ylabel("True positive rate")
#plt.show()
#mean=numpy.mean(pr_list)
#sd=numpy.std(pr_list)
#print(mean,sd)


###############DET CURVE####################################################
'''x_points,y_points=DET_curve_points(posterior_3_dataset,binarized_test_dataset,test_label_dataset,actual_1,actual_8)

plt.plot(y_points,x_points)
plt.xlabel("false positive rate")
plt.ylabel("false negetive rate")
plt.show()'''

##############calculating the equal error rate###############################
min=9999
'''for i in range(0,len(x_points)):
    if abs(x_points[i]-y_points[i])<min:
        err_x=x_points[i]
        err_y=y_points[i]
        min=abs(x_points[i]-y_points[i])
print(err_x,err_y)'''
print(confusion_matrix)
