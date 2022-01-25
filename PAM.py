#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from turbo_code import turbo_code
from AWGN import _AWGN
#import sympy as sym


# In[2]:


class PAM():
  def __init__(self,K,beta1=0.2):
    self.K=K
    self.N=self.K*2
    
    #EbNodB1>EbNodB2
    #User1=Strong User(Fixed)
    #User2=Weak User
    self.EbNodB_diff=10
    
    #Strong Userの受信SNから、βを決定する
    
    self.beta=(beta1**(1/2))/((1-beta1)**(1/2))
    print(self.beta)
    
    self.ch=_AWGN()
    self.cd=turbo_code(self.K)
    
    self.filename="PAM_turbo_BICM_{}_{}_{}".format(self.beta,self.N,self.K)
    
    #self.intleav,self.deintleav=self.interleaver(N)
    
  @staticmethod
  def interleaver(N):
    intleav=np.arange(N)
    #デコーダごとに必要なビットの中でインターリーブする
    #前半と後半ごとにインターリーブする
    np.random.shuffle(intleav[:len(intleav)])
    np.random.shuffle(intleav[len(intleav):])
    deintleav=np.argsort(intleav)
    return intleav,deintleav
  
  def PAM_encode(self):
    info,cwd=self.cd.turbo_encode()
    return info,cwd
  
  def channel(self,cwd,beta):
    #interleave codeword
    #cwd=cwd[self.intleav]
    
    #[cwd1,cwd2]=np.split(cwd,2) #[cwd1,cwd2]
    cwd1=cwd[::2]
    cwd2=cwd[1::2]
    
    const1=self.ch.generate_QAM(cwd1)
    const2=self.ch.generate_QAM(cwd2)
    res_const=beta*const1*(-1*const2)+const2
    #print(res_const)
    return res_const
  
  @staticmethod
  def calc_exp(x,A,No):
    #解が0にならないように計算する
    res=np.zeros(len(x))
    for i in range(len(x)):
      if (x[i]-A)**2/No<30:
        res[i]=np.exp(-1*(x[i]-A)**2/No)
      else:
        res[i]=10**(-15)
    return res
  
  def calc_LLR(self,x,No):
    A1=self.calc_exp(x,-1-self.beta,No)
    A2=self.calc_exp(x,-1+self.beta,No)
    A3=self.calc_exp(x,1-self.beta,No)
    A4=self.calc_exp(x,1+self.beta,No)
    
    y2=np.log((A3+A4)/(A1+A2))
    y1=np.log((A2+A3)/(A1+A4))
    #print(y2)
    #print(y1)
    res=np.zeros(len(y2)+len(y1))
    #res= np.concatenate([y1,y2])
    for i in range(len(y2)):
      res[2*i]=y1[i]
      res[2*i+1]=y2[i]
    
    return res
  
  def decode1(self,res_const,No1):
    RX_const=self.ch.add_AWGN(res_const,No1)
    RX_const=RX_const.real #In-paseのみ取り出す
    #print(RX_const)
    Lc=-1*self.calc_LLR(RX_const,No1)
    #print(Lc)
    EST_cwd=self.cd.turbo_decode(Lc)
    
    #de-interleave codeword
    #EST_cwd=EST_cwd[self.deintleav]
    
    EST_cwd1=EST_cwd[:len(EST_cwd)//2]
    return EST_cwd1
    
  def decode2(self,res_const,No2):
    RX_const=self.ch.add_AWGN(res_const,No2)
    RX_const=RX_const.real #In_phaseのみ取り出す
    Lc=-1*self.calc_LLR(RX_const,No2)
    
    EST_cwd=self.cd.turbo_decode(Lc)
    
    #de-interleave codeword
    #EST_cwd=EST_cwd[self.deintleav]
    
    EST_cwd2=EST_cwd[len(EST_cwd)//2:]
    return EST_cwd2
  
  def PAM_decode(self,res_const,No1,No2):
    EST_cwd1=self.decode1(res_const,No1)
    EST_cwd2=self.decode2(res_const,No2)
    
    return EST_cwd1,EST_cwd2
  
  def main_func(self,EbNodB2):
    #make No1 and No2
    EbNodB1=EbNodB2+self.EbNodB_diff
    EbNo1 = 10 ** (EbNodB1 / 10)
    No1=1/EbNo1
    
    EbNo2 = 10 ** (EbNodB2 / 10)
    No2=1/EbNo2
    
    info,cwd=self.PAM_encode()
    
    res_const=self.channel(cwd,self.beta)
    
    
    EST_info1,EST_info2=self.PAM_decode(res_const,No1,No2)

    EST_info=np.concatenate([EST_info1,EST_info2])
    
    return info,EST_info


# In[6]:


if __name__=="__main__":
  ma=PAM(512)
  a,b=ma.main_func(1)
  #print(a)
  #print(b)
  print(np.sum(a!=b))
  print(a!=b)

