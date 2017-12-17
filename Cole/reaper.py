#coding:utf-8
__author__='lhq'

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

length=500 #油菜田长度为100米
position=0 #当前位置
position_list=[]
time=0
scale=0.6 #仿真以0.6s为间隔
M=4000 #收割机重量4000KG
preSight=5 #提前2米预测作物生长密度
fead=0#喂入量
fead_list=[]
feadOptim=2 #最佳喂入量为2KG/S
timeSight=1 #每隔1s测量一次作物密度
# densityWidth=[0.5,0.05] #假设作物生长密度在服从均值为0.5，方差为0.1的正态分布
# sampleNum=12000 #采样1000个点的作物生长情况
# np.random.seed(123)
# density=np.random.normal(densityWidth[0],densityWidth[1],sampleNum)

x=np.linspace(-2*np.pi,2*np.pi,510000)
density=0.21*np.sin(x)+0.5  #取正态分布不合理，因为田间作物密度不是突变的

density_pre=0 #实时检测作物密度
density_cur=0
v=4 #记录速度
v_list=[]
preV=0 #预测的速度
a=0
max_a=0.4 #最大的加速度为0.1m/s^2
min_a=-0.3
max_v=8 #最大速度为8m/s
min_v=0
fo=[]
den=[]
prePositon=2
base=0
base_list=[]

def clip(num,max_num,min_num):
    if(num>max_num):
        num=max_num
    elif(num<min_num):
        num=min_num
    else:
        num=num
    return num

v_list2=[]
count=0
v2=4
a2=0
position_list2=[]
a_list=[]
# 不考虑常量，作物密度与前进速度均与喂入量成线性关系，f=dv
while(position+preSight<length):

    density_cur=density[int(np.around(position)*1000)] #当前作物密度
    # preV=clip(preV,max_v,min_v)
    fead=v*density_cur
    fead_list.append(fead)
    if(time%0.01<1e-2): #每隔1s检测一次
        density_pre=density[int(np.around(position+preSight)*1000)] #获取2米外的作物密度
        preV=feadOptim/density_pre
        # preV=clip(preV,max_v,min_v)
        prePositon=position+preSight
        a=(preV**2-v**2)/(2*(prePositon-position)) #计算当前加速度


        a=clip(a,max_a,min_a)



    v += a*scale
    v=clip(v,max_v,min_v)



    position += v*scale+(a*scale**2)/2 #计算当前位置
    time += scale

    v_list.append(v)
    position_list.append(position)
    fo.append(feadOptim)
    den.append(density_cur)
    base=4*density_cur
    base_list.append(base)
    count += 1
print(count)

position=0
fead_list2=[]
deltav=0

while(position<length):

    time=0
    density_cur = density[int(np.around(position) * 1000)]  # 当前作物密度
    # preV=clip(preV,max_v,min_v)
    fead = v2 * density_cur
    fead_list2.append(fead)
    if (time % 0.01 < 1e-3):
        deltav=(feadOptim-fead)/density_cur
        a2=deltav/scale
        a2=clip(a2,max_a,min_a)

    v2 += a2*scale
    v2=clip(v2,max_v,min_v)
    v_list2.append(v2)
    position += v2 * scale + (a2 * scale ** 2) / 2  # 计算当前位置
    time += scale
    position_list2.append(position)
    print(position)


from scipy.interpolate import spline

position_list=np.array(position_list)
fead_list=np.array(fead_list)
base_list=np.array(base_list)
ax=plt.subplot(111)
ax.set_xlabel('Position',fontsize=20)
ax.set_ylabel('Fead quantity',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
xnew = np.linspace(position_list.min(),position_list.max(),1000)
# smooth1 = spline(position_list,fead_list,xnew)
# smooth2 = spline(position_list,base_list,xnew)


plt.plot(position_list,fead_list,'r',label='Fead quantity by pre-control',ls='--',linewidth=4.0)

# plt.plot(position_list,v_list2,'r')
plt.plot(position_list2,fead_list2,'b',label='Fead quantity by measurement-control',ls=':',linewidth=4.0)
# plt.plot(position_list,den,'g')
plt.plot(position_list,base_list,'g',label='Fead quantity without control',ls='-.',linewidth=4.0)
plt.plot(position_list,fo,'k',label='Optimal fead quantity')

# plt.plot(position_list2,fead_list2,'b')
plt.axis([0,510,0,4])
plt.legend(fontsize=20)

plt.show()
