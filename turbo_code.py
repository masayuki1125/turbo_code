#!/usr/bin/env python
# coding: utf-8

# In[26]:

from AWGN import _AWGN
import numpy as np
ch=_AWGN()

# In[27]:

class coding():

  def __init__(self,K):
    super().__init__()

    # to use in class:turbo_code
    self.numerator=np.array([1,0,0])
    self.denominator=np.array([1,0,1])
    self.K=K
    self.max_itr=8

    #check
    if len(self.numerator)!=len(self.denominator):
      print("please set the same length between numerator and denominator!")
      exit()

    #set constant
    self.interleaver_sequence,self.de_interleaver_sequence=self.interleave()
    self.T,self.G=self.trellis()

    #to write txt file
    self.R=str(1)+"|"+str(3)# use later
    self.filename="turbo_code_{}_{}".format(self.K,self.R)

  def interleave(self):
    interleaver_sequence=np.arange(self.K)
    np.random.shuffle(interleaver_sequence)
    de_interleaver_sequence=np.argsort(interleaver_sequence)
    return interleaver_sequence,de_interleaver_sequence

  @staticmethod
  def binary(x,memory_num):
    res=np.zeros((memory_num),dtype=int)
    for i in range(memory_num):
      res[memory_num-i-1]=x%2
      x=x//2

    return res
  
  @staticmethod
  def decimal(memory):
    res=0
    for i in range(len(memory)):
      res=res+memory[i]*(2**(len(memory)-i-1))

    return res

  def trellis(self):
    memory_num=len(self.numerator)-1
    T=[[],[]]
    G=np.zeros((2,2**memory_num,2**memory_num))
    
    #make T and G
    for j in ([0,1]):  
      for i in range(2**memory_num):
        memory=self.binary(i,memory_num)

        #print(memory)
        information=np.array([j])
        parity,memory=self.IIR_encoder_for_trellis(information,memory)

        #print(memory)
        T[j]=T[j]+[(i,self.decimal(memory))]
        G[0,i,self.decimal(memory)]=2*j-1
        G[1,i,self.decimal(memory)]=2*parity-1

    return T,G

  def IIR_encoder_for_trellis(self,information,memory):
  
    tmp=np.zeros(len(memory)+1,dtype=int)
    parity=np.zeros(len(information))

    for i,m in enumerate(information):

      #step1 calculate denominator and store deno_res
      tmp[0]=m
      tmp[1:]=memory
      deno_res=np.sum(tmp*self.denominator)%2
      #print(deno_res)

      #step2 calculate numerator and generate codeword
      tmp[0]=deno_res
      num_res=np.sum(tmp*self.numerator)%2
      #print(num_res)
      memory[:]=tmp[:len(tmp)-1]

      parity[i]=num_res

    return parity,memory

# In[37]:

class encoding(coding):

  def __init__(self,K):
    super().__init__(K)

  def generate_information(self):
    #generate information
    information=np.random.randint(0,2,self.K)
    return information

  def IIR_encoder(self,information):

    #initiarize memory
    memory=np.zeros([len(self.numerator)-1],dtype=int)
  
    tmp=np.zeros(len(memory)+1,dtype=int)
    parity=np.zeros(len(information))

    for i,m in enumerate(information):

      #step1 calculate denominator and store deno_res
      tmp[0]=m
      tmp[1:]=memory
      deno_res=np.sum(tmp*self.denominator)%2
      #print(deno_res)

      #step2 calculate numerator and generate codeword
      tmp[0]=deno_res
      num_res=np.sum(tmp*self.numerator)%2
      #print(num_res)
      memory[:]=tmp[:len(tmp)-1]

      parity[i]=num_res

    return parity

  def turbo_encode(self):

    information=self.generate_information()
    
    codeword=np.zeros(3*len(information))
    codeword[::3]=information

    codeword[1::3]=self.IIR_encoder(information)
    codeword[2::3]=self.IIR_encoder(information[self.interleaver_sequence])

    return information,codeword

# In[51]:

class decoding(coding):

  @staticmethod
  def maxstr(x):
    tmp=np.max(x)
    x=x-tmp
    res=tmp+np.log(1+np.sum(np.exp(x)))
    return res

  def BCJR(self,lambda_s,lambda_p,lambda_e):

    #prepere matrices
    NO_PATH=-10.0**100
    log_gamma=np.full((len(lambda_s),self.G[0].shape[0],self.G[0].shape[1]),NO_PATH)
    log_alpha=np.full((len(lambda_s),self.G[0].shape[0]),NO_PATH)
    log_beta=np.full(log_alpha.shape,NO_PATH)
    res=np.zeros((len(lambda_s)))

    #set initial state
    log_alpha[0,0]=0.0
    log_beta[len(lambda_s)-1]=np.log(1/(self.G[0].shape[0]))

    #culculate gamma
    #print(lambda_e)
    for i in range(len(lambda_s)):
      log_gamma[i]=1/2*self.G[0]*lambda_s[i]+1/2*self.G[1]*lambda_p[i]+1/2*self.G[0]*lambda_e[i]

    log_gamma[log_gamma==0]=NO_PATH
    #print("gamma=",log_gamma)
    #print(log_gamma[0,0,:])

    #calculate alpha
    for i in range(len(lambda_s)-1):
      for j in range(log_alpha.shape[1]):
        log_alpha[i+1,j]=self.maxstr(log_alpha[i]+log_gamma[i,:,j])

    #print("alpha=",log_alpha)

    #calculate beta
    for i in reversed(range(len(lambda_s)-1)):
      for j in range(log_beta.shape[1]):
        log_beta[i,j]=self.maxstr(log_beta[i+1]+log_gamma[i+1,j])

    #print("beta=",log_beta)

    #calculate llr
    for i in range(len(lambda_s)):
      # set s=0 or s=1 branch metricc
      tmp=np.zeros((len(self.T[0]),2))
      for j in range(tmp.shape[0]):
        for k in ([0,1]):
          tmp[j,k]=log_alpha[i,j]           +log_gamma[i,j,self.T[k][j][1]]+log_beta[i,self.T[k][j][1]]
      res[i]=self.maxstr(tmp[:,1])-self.maxstr(tmp[:,0])

    return res

  def turbo_decode(self,Lc,max_itr):
    lambda_s,lambda_p1,lambda_p2=Lc[::3],Lc[1::3],Lc[2::3]

    itr=0
    lambda_e=np.zeros((len(lambda_s)))
    
    while itr<max_itr:
        
      #first decoder
      res=self.BCJR(lambda_s,lambda_p1,lambda_e)
      #print(lambda_e)
      lambda_e=res-lambda_s-lambda_e

      #second decoder
      lambda_s=lambda_s[self.interleaver_sequence]
      lambda_e=lambda_e[self.interleaver_sequence]

      res=self.BCJR(lambda_s,lambda_p2,lambda_e)

      lambda_e=res-lambda_s-lambda_e

      lambda_s=lambda_s[self.de_interleaver_sequence]
      lambda_e=lambda_e[self.de_interleaver_sequence]

      itr+=1

    res=res[self.de_interleaver_sequence]
    res=np.sign(res)
    EST_information=(res+1)/2

    return EST_information

# In[63]:

class turbo_code(encoding,decoding):
  def __init__(self,K):
    super().__init__(K)
    
  def main_func(self,EbNodB): 
    information,codeword=self.turbo_encode()
    Lc=ch.generate_LLR(codeword,EbNodB)#デコーダが＋、ー逆になってしまうので-１をかける
    EST_information=self.turbo_decode(Lc,self.max_itr)      
    return information,EST_information

# In[88]:

if __name__=="__main__":
  tc=turbo_code(100)
  def output(EbNodB):

    #prepare some constants
    MAX_ERR=8
    count_ball=0
    count_berr=0
    count_all=0
    count_err=0
    

    while count_err<MAX_ERR:
    #print("\r"+str(count_err),end="")
      information,EST_information=tc.main_func(EbNodB)
      
      #calculate block error rate
      if np.any(information!=EST_information):
          count_err+=1
      count_all+=1

      #calculate bit error rate 
      count_berr+=np.sum(information!=EST_information)
      count_ball+=len(information)

      print("\r","count_all=",count_all,",count_err=",count_err,"count_ball=",count_ball,"count_berr=",count_berr,end="")

    return count_err,count_all,count_berr,count_ball
  
  output(-1)
