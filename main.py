#######################################################
### Code forLine Fitting                           ####
###                                                ####
### Submitted at university of Marlynd, ENPM673    ####
###                                                ####
### Developed by - Yashaarth Todi                  ####
#######################################################
import numpy as np
import pickle
import matplotlib.pyplot as plt

fileread=open("data1_new.pkl","rb")
d1=pickle.load(fileread)
matrix=np.mat(d1)
a1=np.array(d1)

#Finding the covariance matrix
tm=np.transpose(matrix)
c=np.cov(tm)
print("Covariance matrix: \n",c)

#Finding eigen values and vectors of Covariance Matrix
evalues,evectors=np.linalg.eig(c)
evalues=np.mat(evalues)
evectors=np.mat(evectors)
print("\n Eigen values are: \n",evalues)
print("\n Eigen vectors are: \n",evectors)

#Creating array for x and y data points
xs = [x[0] for x in d1]
ys = [x[1] for x in d1]

#Finding the slope of Eigenvector1 and Eigenvector2
if evalues[0,0]>evalues[0,1]:
    m1 = evectors[1,0]/evectors[0,0]
    m2 = evectors[1, 1] / evectors[0, 1]
else:
    m1 = evectors[1,1]/evectors[0,1]
    m2= evectors[1,0]/evectors[0,0]

vpy1=np.array([])
vpy2=np.array([])

#Forming the data set for plotting Eigenvectors
for i in range(0,len(a1)):
    v1=m1*a1[i,0]
    vpy1=np.append(vpy1,v1)
    v2=m2*a1[i,0]
    vpy2=np.append(vpy2,v2)
length=len(xs)

#Geometric Interpretation of Covariance Matrix
L=np.diag(evalues.A1)
SL=np.sqrt(L)
T=np.dot(evectors,SL)
print("\n The Transformation matrix is: \n",T)
tdata=np.matmul(T,np.transpose(d1))

tdata=np.transpose(tdata)
x=np.array(tdata[:,0])
y=np.array(tdata[:,1])
y1=np.dot(m1,x)
y2=np.dot(m2,x)

# Line Fitting using Vertical Distance
Y=np.transpose(ys)
X1=np.ones((length))
ty=np.array(xs).reshape(200,1)
X1=np.array(X1).reshape(200,1)
X=np.hstack((ty,X1))
s1=np.linalg.inv(np.matmul(np.transpose(X),X))
s2=np.matmul(np.transpose(X),(Y))
s2=np.transpose(s2)
ans=np.matmul(s1,s2)
yls=np.dot(ans[0],xs)+ ans[1] #yls is the line for Ordinary least sqaures ( Vertical dist)

# Line Fitting using Orthogonal Distance
meanx=np.average(matrix[:,0])
meany=np.average(matrix[:,1])
u=np.column_stack((xs-meanx,ys-meany))
n= np.matmul(np.transpose(u),u)

#Finding Eigenvalues and Eigenvectors for PCA
evalues1,evectors1=np.linalg.eig(n)
evalues1=np.mat(evalues1)
evectors1=np.mat(evectors1)
print("\n Eigen values are: \n",evalues1)
print("\n Eigen vectors are: \n",evectors1)

#Finding the Eigen vector corresponding to smallest eigen value
if evalues1[0,0]>evalues1[0,1]:
    vector=evectors1[:,1]
else:
    vector=evectors1[:,0]
print("Smallest eigen vector are \n",vector)
m3=vector[1]/vector[0]
m33=-1/m3
c=meany-m33*meanx
y3=m33*xs+c

xs=np.array(xs)
ys=np.array(ys)
s=2
e=0.5
count=0
#RANSAC Implementation
N=(np.log10(1-0.99))/(np.log10(1-(1-e)**s))
print("RANSAC ")
bestinlier=0
while N>count:
    randpoints=np.random.permutation(d1)[:2] #Choosing two random points
    mrand=(randpoints[0,1]-randpoints[1,1])/(randpoints[0,0]-randpoints[1,0]) #Model building
    crand=randpoints[0,1]-mrand*randpoints[0,0]
    drand=abs(ys-mrand*xs-crand)/(np.sqrt(1+mrand**2)) #Distance of points from model
    variance=np.mean(drand**2)-(np.mean(drand))**2
    tsqaure=3.84*variance
    t=np.sqrt(tsqaure) #Deciding acceptable threshold for inliers
    j=0
    filterdata=[]

    indexlist = [] # Array with indexes of inliers
    for i in range(0, len(drand)):
        if drand[i] < t:
            indexlist.insert(j, i)
            filterdata.insert(j,d1[i])
            j = j + 1
    inliers = len(indexlist) # Counting nummber of inlier points
    if bestinlier<inliers:
        bestinlier=inliers #Updating max inliers
        mransac=mrand       # Updating best model
        cransac=crand
    e = 1 - (inliers /length) # Calculating Outlier ratio
    N=(np.log10(1-0.99))/(np.log10(1-(1-e)**s)) # Calculating number of iterations
    count=count+1
print("Number of inliers are",bestinlier,"out of",length)

#Applying Least sqaures to refine RANSAC data
filterdata=np.array(filterdata)
filterx = [x[0] for x in filterdata]
filtery = [x[1] for x in filterdata]
newmeanx=np.average(filterx)
newmeany=np.average(filtery)
unew=np.column_stack((filterx-newmeanx,filtery-newmeany))
newn= np.matmul(np.transpose(unew),unew)
evalues2,evectors2=np.linalg.eig(newn)
evalues2=np.mat(evalues2)
evectors2=np.mat(evectors2)
if evalues2[0,0]>evalues2[0,1]:
    vector2=evectors2[:,1]
else:
    vector2=evectors2[:,0]

newm3=vector2[1]/vector2[0]
newm33=-1/newm3
newc=newmeany-newm33*newmeanx
newy3=newm33*filterx+newc
fig,ax=plt.subplots()
#Plotting Part-1 (Eigen Vectors)
plt.figure(1)
plt.scatter(xs,ys,color='black')
plt.plot(xs,vpy1,linestyle='-', label='First Eigen vector',color='red')
plt.plot(xs,vpy2,linestyle='-', label='Second Eigen vector',color='blue')
legend = ax.legend(loc='upper center', shadow=True, fontsize='large')

fig,ax=plt.subplots()
#Plotting Part-2 (Least squares)
plt.figure(2)
#fig,ax=plt.subplots()
plt.scatter(xs,ys,color='black')
plt.plot(xs,yls,linestyle='-', label='Least Squares- Vertical Distance',color="yellow") #Least squares- vertical distance
plt.plot(xs,np.transpose(y3),linestyle='-', label='Least Squares-Perpendicular Distance',color="red") #Least squares- Perpendicular distance
legend = ax.legend(loc='upper center', shadow=True, fontsize='large')

fig,ax=plt.subplots()
#Plotting part 3-RANSAC
plt.figure(3)
plt.scatter(xs,ys,color='yellow')
plt.plot(xs,mransac*xs+cransac,linestyle='-', label='RANSAC',color="blue") #RANSAC
plt.plot(filterx,np.transpose(newy3),linestyle='-', label=' RANSAC+Least fit',color="green")
legend = ax.legend(loc='upper center', shadow=True, fontsize='large')
plt.show()
