#!/usr/bin/env python
# coding: utf-8

# In[75]:


from AWGN import _AWGN
import numpy as np
ch=_AWGN()


# In[ ]:


class coding():

  def __init__(self,K):
    super().__init__()

    # to use in class:turbo_code
    #[1,x,x^2,x^3,....]
    self.numerator=np.array([1,0])
    self.denominator=np.array([1,1])
    self.K=K
    self.max_itr=18

    #check
    if len(self.numerator)!=len(self.denominator):
      print("please set the same length between numerator and denominator!")
      exit()

    #set constant
    self.interleaver_sequence,self.de_interleaver_sequence=self.interleave()

    self.G=self.trellis()

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
    
    #G=np.full((2,2**memory_num,2**memory_num),np.nan)
    G=np.zeros((2,2**memory_num,2**memory_num))
    
    #make G
    for j in ([0,1]):  
      for i in range(2**memory_num):

        #iを2進数に変更
        memory=self.binary(i,memory_num)

        #print(memory)
        information=np.array([j])
        parity,memory=self.IIR_encoder_for_trellis(information,memory)

        #print(memory)
        G[0,i,self.decimal(memory)]=2*j-1
        G[1,i,self.decimal(memory)]=2*parity-1

    return G

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


# In[88]:


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

      state=self.decimal(memory)

    return parity,state

  def turbo_encode(self):

    while True:
      information=self.generate_information()
      #information=np.array([1,1,0,0,1,0,1,0,1,1])
      
      codeword=np.zeros(3*len(information))
      codeword[::3]=information

      codeword[1::3],state1=self.IIR_encoder(information)
      codeword[2::3],state2=self.IIR_encoder(information[self.interleaver_sequence])

      if state1==0 and state2==0:
        break

    return information,codeword


# In[91]:


class decoding(coding):
  
  @staticmethod
  def maxstr(x1,x2=np.nan):
    UPPER_THRES=500
    if np.isnan(x2):
      return x1
    else:
      tmp1=max(x1,x2)
      tmp2=abs(x1-x2)
      if tmp2>UPPER_THRES:
        res=tmp1 #only max operation
      else:
        res=tmp1+np.log(1+np.exp(-1*tmp2))
      return res

  def BCJR(self,lambda_s,lambda_p,lambda_pri):

    #prepere matrices
    log_parity=np.full((len(lambda_s),self.G[0].shape[0],self.G[0].shape[1]),np.nan)
    log_gamma=np.full((len(lambda_s),self.G[0].shape[0],self.G[0].shape[1]),np.nan)
    log_alpha=np.full((len(lambda_s),self.G[0].shape[0]),np.nan)
    log_beta=np.full(log_alpha.shape,np.nan) # the same as beta
    res=np.full((len(lambda_s)),np.nan) #結果のLLRを格納する配列

    lambda_e=np.zeros(len(lambda_s))

    #set initial state
    log_alpha[0,0]=0
    log_beta[len(lambda_s)-1,0]=0

    #culculate gamma
    for i in range(len(lambda_s)):
      log_parity[i]=1/2*lambda_p[i]*self.G[1]
      log_gamma[i]=1/2*lambda_s[i]*self.G[0]+1/2*lambda_pri[i]*self.G[0]+log_parity[i]
    #print(log_gamma)
    
    ##calculate alpha
    for i in range(len(lambda_s)-1):
      for k in range(self.G[0].shape[0]): #縦列ごとに見ていく
        tmp=np.nan
        for j in range(self.G[0].shape[1]):
          if np.isnan(log_gamma[i,j,k])==False and np.isnan(log_alpha[i,j])==False: #path is existing or not
            log_alpha[i+1,k]=self.maxstr(log_alpha[i,j]+log_gamma[i,j,k],tmp)
            tmp=log_alpha[i+1,k]
    #check alpha
    #np.savetxt("alpha",log_alpha)
    #from IPython.core.debugger import Pdb; Pdb().set_trace()

    ##calculate beta
    for i in reversed(range(len(lambda_s)-1)):
      for j in range(self.G[0].shape[0]): #横列ごとに見ていく
        tmp=np.nan
        for k in range(self.G[0].shape[1]):
          if np.isnan(log_gamma[i+1,j,k])==False and np.isnan(log_beta[i+1,k])==False:
            log_beta[i,j]=self.maxstr(log_beta[i+1,k]+log_gamma[i+1,j,k],tmp)
            tmp=log_beta[i,j]
    #check beta
    #np.savetxt("beta",log_beta)
    #from IPython.core.debugger import Pdb; Pdb().set_trace()

    #それぞれのpathについて考える
    for i in range(len(lambda_s)):
      tmp_plus=np.nan #＋のpathを総てmaxstrした値を格納
      tmp_minus=np.nan #-のpathを総てmaxstrした値を格納
      #総てのpathについて考える
      for j in range(self.G[0].shape[0]):
        for k in range(self.G[0].shape[1]):
          if np.isnan(log_gamma[i,j,k])==False and np.isnan(log_alpha[i,j])==False and np.isnan(log_beta[i,k])==False: #pathが存在するとき
            if self.G[0,j,k]==-1:#info_bit=0のパスだったとき   
                tmp_minus=self.maxstr(log_alpha[i,j]+log_beta[i,k]+log_parity[i,j,k],tmp_minus)
            elif self.G[0,j,k]==1:#info_bit=1のパスだったとき
                tmp_plus=self.maxstr(log_alpha[i,j]+log_beta[i,k]+log_parity[i,j,k],tmp_plus)
      #from IPython.core.debugger import Pdb; Pdb().set_trace()

      lambda_e[i]=tmp_plus-tmp_minus
      #ckeck
      #if lambda_e[i]==0:
        #print("error")
        #print(lambda_e[i])
        #print(log_alpha[i])
        #print(log_beta[i])
        #print(log_gamma[i])
    
    #return LLR
    res=lambda_e+lambda_s+lambda_pri

    return res,lambda_e

  def turbo_decode(self,Lc,max_itr):
    lambda_s,lambda_p1,lambda_p2=Lc[::3],Lc[1::3],Lc[2::3]
    in_lambda_s=lambda_s[self.interleaver_sequence]

    #punctureing(optional)
    #lambda_p1[0::2]=0
    #lambda_p2[1::2]=0

    itr=0
    lambda_e=np.zeros((len(lambda_s)))
    
    while itr<max_itr:
          
      #first decoder
      _,lambda_e=self.BCJR(lambda_s,lambda_p1,lambda_e[self.de_interleaver_sequence])
      #print(lambda_e)

      #second decoder
      res,lambda_e=self.BCJR(in_lambda_s,lambda_p2,lambda_e[self.interleaver_sequence])
      #print(lambda_e)
      
      itr+=1
    
    res=res[self.de_interleaver_sequence]
    res=np.sign(res)
    EST_information=(res+1)/2

    return EST_information


# In[94]:


class turbo_code(encoding,decoding):
  def __init__(self,K):
    super().__init__(K)
    
  def main_func(self,EbNodB): 
    information,codeword=self.turbo_encode()
    Lc=ch.generate_LLR(codeword,EbNodB)
    EST_information=self.turbo_decode(Lc,self.max_itr)      
    return information,EST_information


# In[95]:


if __name__=="__main__":
  tc=turbo_code(10000)
  print(tc.G)
  def output(EbNodB):

    #prepare some constants
    MAX_ERR=8
    count_ball=0
    count_berr=0
    count_all=0
    count_err=0
    #count_berr_mat=np.zeros(1000)

    while count_err<MAX_ERR:
    #print("\r"+str(count_err),end="")
      information,EST_information=tc.main_func(EbNodB)
      
      #calculate block error rate
      if np.any(information!=EST_information):
          count_err+=1
      count_all+=1

      #calculate bit error rate 
      count_berr+=np.sum(information!=EST_information)
      #count_berr_mat+=information!=EST_information
      #print(count_berr_mat)
      count_ball+=len(information)

      print("\r","count_all=",count_all,",count_err=",count_err,"count_ball=",count_ball,"count_berr=",count_berr,end="")

    return count_err,count_all,count_berr,count_ball

  EbNodB_range=np.array([-4.776,-3.776,-2.776,-1.776])
  for i in EbNodB_range:
    print(i)
    _,_,a,b=output(i)
    print(a/b)

