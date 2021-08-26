#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append("../channel")
from AWGN import _AWGN
import numpy as np
from fractions import Fraction
ch=_AWGN()


# In[3]:


class coding():

  def __init__(self,K=100):
    super().__init__()

    # to use in class:turbo_code
    self.numerator=np.array([1,0,0])
    self.denominator=np.array([1,0,1])
    self.K=K
    self.L_MAX=8

    #set constant
    self.interleaver_sequence,self.de_interleaver_sequence=interleave(self.K)
    self.T,self.G=trellis(self.numerator,self.denominator)

    #to write txt file
    self.R=str(1)+"|"+str(3)# use later
    self.filename="turbo_code_{}_{}".format(self.K,self.R)


# In[4]:


def generate_information(K):
  #generate information
  information=np.random.randint(0,2,K)
  return information


# In[5]:


def interleave(K):
  interleaver_sequence=np.arange(K)
  np.random.shuffle(interleaver_sequence)
  de_interleaver_sequence=np.argsort(interleaver_sequence)
  return interleaver_sequence,de_interleaver_sequence


# In[6]:


def IIR_encoder(information,numerator,denominator,memory):

  if len(numerator)!=len(denominator):
    print("please set the same length between numerator and denominator!")
    exit()
 
  tmp=np.zeros(len(memory)+1,dtype=int)
  parity=np.zeros(len(information))

  for i,m in enumerate(information):

    #step1 calculate denominator and store deno_res
    tmp[0]=m
    tmp[1:]=memory
    deno_res=np.sum(tmp*denominator)%2
    #print(deno_res)

    #step2 calculate numerator and generate codeword
    tmp[0]=deno_res
    num_res=np.sum(tmp*numerator)%2
    #print(num_res)
    memory[:]=tmp[:len(tmp)-1]

    parity[i]=num_res

  return parity,memory


# In[7]:


def binary(x,memory_num):
  res=np.zeros((memory_num),dtype=int)
  for i in range(memory_num):
      res[memory_num-i-1]=x%2
      x=x//2

  return res

def decimal(memory):
  res=0
  for i in range(len(memory)):
      res=res+memory[i]*(2**(len(memory)-i-1))
  
  return res

def trellis(numerator,denominator):
  memory_num=len(numerator)-1
  T=[[],[]]
  G=np.zeros((2,2**memory_num,2**memory_num))
  #make T and G
  for j in ([0,1]):      
    for i in range(2**memory_num):
        memory=binary(i,memory_num)
        #print(memory)
        information=np.array([j])
        parity,memory=IIR_encoder(information,numerator,denominator,memory)
        #print(memory)
        T[j]=T[j]+[(i,decimal(memory))]
        G[0,i,decimal(memory)]=2*j-1
        G[1,i,decimal(memory)]=2*parity-1

  return T,G


# In[8]:


def turbo_encoder(information,interleaver_sequence,numerator,denominator):

  codeword=np.zeros(3*len(information))
  codeword[::3]=information

  memory=np.zeros([len(numerator)-1],dtype=int)
  codeword[1::3],_=IIR_encoder(information,numerator,denominator,memory)
  memory=np.zeros([len(numerator)-1],dtype=int)
  codeword[2::3],_=IIR_encoder(information[interleaver_sequence],numerator,denominator,memory)

  return codeword


# In[9]:


def maxstr(x):
  tmp=np.max(x)
  x=x-tmp
  res=tmp+np.log(1+np.sum(np.exp(x)))
  return res

def BCJR(lambda_s,lambda_p,lambda_e,T,G):

  #prepere matrices
  NO_PATH=-10.0**100
  log_gamma=np.full((len(lambda_s),G[0].shape[0],G[0].shape[1]),NO_PATH)
  log_alpha=np.full((len(lambda_s),G[0].shape[0]),NO_PATH)
  log_beta=np.full(log_alpha.shape,NO_PATH)
  res=np.zeros((len(lambda_s)))

  #set initial state
  log_alpha[0,0]=0.0
  log_beta[len(lambda_s)-1]=np.log(1/(G[0].shape[0]))

  #culculate gamma
  #print(lambda_e)
  for i in range(len(lambda_s)):
    log_gamma[i]=1/2*G[0]*lambda_s[i]+1/2*G[1]*lambda_p[i]+1/2*G[0]*lambda_e[i]

  log_gamma[log_gamma==0]=NO_PATH
  #print("gamma=",log_gamma)
  #print(log_gamma[0,0,:])

  #calculate alpha
  for i in range(len(lambda_s)-1):
    for j in range(log_alpha.shape[1]):
      log_alpha[i+1,j]=maxstr(log_alpha[i]+log_gamma[i,:,j])

  #print("alpha=",log_alpha)

  #calculate beta
  for i in reversed(range(len(lambda_s)-1)):
    for j in range(log_beta.shape[1]):
      log_beta[i,j]=maxstr(log_beta[i+1]+log_gamma[i+1,j])

  #print("beta=",log_beta)

  #calculate llr
  for i in range(len(lambda_s)):
    # set s=0 or s=1 branch metric
    tmp=np.zeros((len(T[0]),2))
    for j in range(tmp.shape[0]):
      for k in ([0,1]):
        tmp[j,k]=log_alpha[i,j]         +log_gamma[i,j,T[k][j][1]]+log_beta[i,T[k][j][1]]
    res[i]=maxstr(tmp[:,1])-maxstr(tmp[:,0])

  return res


# In[10]:


def turbo_decode(lambda_s,lambda_p1,lambda_p2,T,G,interleaver_sequence,de_interleaver_sequence,max_itr=8):

  itr=0
  lambda_e=np.zeros((len(lambda_s)))
  while itr<max_itr:
      
      #first decoder
      res=BCJR(lambda_s,lambda_p1,lambda_e,T,G)
      #print(lambda_e)
      lambda_e=res-lambda_s-lambda_e

      #second decoder
      lambda_s=lambda_s[interleaver_sequence]
      lambda_e=lambda_e[interleaver_sequence]

      res=BCJR(lambda_s,lambda_p2,lambda_e,T,G)

      lambda_e=res-lambda_s-lambda_e

      lambda_s=lambda_s[de_interleaver_sequence]
      lambda_e=lambda_e[de_interleaver_sequence]

      itr+=1

  res=res[de_interleaver_sequence]

  return res


# In[11]:


class turbo_code(coding):
  def __init__(self,K):
    super().__init__(K)
  def encode(self):
    print("turbo_code uses",self.K)
    information=generate_information(self.K)
    #information=np.zeros(self.K)#check
    codeword=turbo_encoder(information,self.interleaver_sequence,self.numerator,self.denominator)
    return codeword,information

  def decode(self,r0,r1,r2):
    res=turbo_decode(r0,r1,r2,self.T,self.G,self.interleaver_sequence,self.de_interleaver_sequence,self.L_MAX)
    res=np.sign(res)
    res=(res+1)/2
    return res

  def turbo_code(self,EbNodB): 
    codeword,information=self.encode()
    Lc=ch.generate_LLR(codeword,EbNodB)
    r0,r1,r2=Lc[::3],Lc[1::3],Lc[2::3]
    EST_information=self.decode(r0,r1,r2)
    return information,EST_information


# In[13]:

'''
if __name__=="__main__":
    K=[200]
    for K in K:
        print("K=",K)
        tc=turbo_code(K)
        #st=savetxt()
        #BLER,BER=mc.monte_carlo(tc.turbo_code)
        #st.savetxt(BLER,BER)
'''
