#!/usr/bin/env python
# coding: utf-8

# In[104]:


# -*- coding: utf-8 -*-
"""
Created on 21 Aug 2014
Functions and classes for convolutional and turbo codes.
Author: Venkat Venkatesan
from https://github.com/venkat0791/codec
"""
import math
import numpy as np
from AWGN import _AWGN
ch=_AWGN()


# In[105]:


class conv_code():
  INF = 1e6

  def __init__(self, back_poly, fwd_polys):

    # Number of bits in encoder state.
    self.mem_len = math.floor(math.log(back_poly) / math.log(2))

    # Encoder state space (integers in the range [0, 2 ** mem_len)).
    self.state_space = tuple(n for n in range(1 << self.mem_len))

    # Number of encoder output bits per input bit.
    self.n_out = len(fwd_polys)

    # MSB of next encoder state, given current state and input bit.
    self.next_state_msb = tuple(tuple(self.bitxor(back_poly & ((b << self.mem_len) + s))for s in self.state_space) for b in (0, 1))

    # Encoder output bits, given current state and input bit.
    self.out_bits = tuple(tuple(tuple(self.bitxor(p & ((self.next_state_msb[b][s] << self.mem_len) + s))for p in fwd_polys) for s in self.state_space) for b in (0, 1))
    
    # Next encoder state, given current state and input bit.
    self.next_state = tuple(tuple((self.next_state_msb[b][s] << (self.mem_len - 1)) + (s >> 1)for s in self.state_space) for b in (0, 1))

    return


# In[106]:


class conv_code(conv_code):

  @staticmethod
  def bitxor(num):
    '''
    Returns the XOR of the bits in the binary representation of the
    nonnegative integer num.
    '''

    count_of_ones = 0
    while num > 0:
      count_of_ones += num & 1
      num >>= 1

    return count_of_ones % 2

  @staticmethod
  def maxstar(eggs, spam, max_log=True):
    '''
    Returns log(exp(eggs) + exp(spam)) if not max_log, and max(eggs, spam)
    otherwise.
    '''
    return max(eggs, spam) + (0 if max_log else math.log(1 + math.exp(-abs(spam - eggs))))


# In[107]:


class conv_code(conv_code):

  def encode(self, info_bits):

    #info_bits = np.asarray(info_bits).ravel()
    n_info_bits = info_bits.size

    code_bits, enc_state = -np.ones(self.n_out * (n_info_bits + self.mem_len), dtype=int), 0
    
    for k in range(n_info_bits + self.mem_len):
      in_bit = (info_bits[k] if k < n_info_bits else self.next_state_msb[0][enc_state])
      #print((self.out_bits[in_bit][enc_state]))
      code_bits[self.n_out * k : self.n_out * (k + 1)] = (self.out_bits[in_bit][enc_state])
      enc_state = self.next_state[in_bit][enc_state]

    return code_bits


# In[108]:


class conv_code(conv_code):

  def _branch_metrics(self, out_bit_llrs, pre_in_bit_llr=0):

    gamma_val = ([pre_in_bit_llr / 2 for s in self.state_space],[-pre_in_bit_llr / 2 for s in self.state_space])
    for enc_state in self.state_space:
      for bit0, bit1, val in zip(self.out_bits[0][enc_state],self.out_bits[1][enc_state],out_bit_llrs):
        gamma_val[0][enc_state] += val / 2 if bit0 == 0 else -val / 2
        gamma_val[1][enc_state] += val / 2 if bit1 == 0 else -val / 2

    return gamma_val

  def _update_path_metrics(self, out_bit_llrs, path_metrics, best_bit):

    gamma_val = self._branch_metrics(out_bit_llrs)

    pmn = path_metrics[:]
    for enc_state in self.state_space:
      cpm0 = gamma_val[0][enc_state] + pmn[self.next_state[0][enc_state]]
      cpm1 = gamma_val[1][enc_state] + pmn[self.next_state[1][enc_state]]
      path_metrics[enc_state], best_bit[enc_state] = ((cpm0, 0) if cpm0 >= cpm1 else (cpm1, 1))

    return


# In[109]:


class conv_code(conv_code):
  def decode_viterbi(self, code_bit_llrs):

    code_bit_llrs = np.asarray(code_bit_llrs).ravel()
    n_in_bits = int(code_bit_llrs.size / self.n_out)
    n_info_bits = n_in_bits - self.mem_len

    # Path metric for each state at time n_in_bits.
    path_metrics = [(0 if s == 0 else -conv_code.INF) for s in self.state_space]

    # Best input bit in each state at times 0 to n_in_bits - 1.
    best_bit = [[-1 for s in self.state_space] for k in range(n_in_bits)]

    # Start at time n_in_bits - 1 and work backward to time 0, finding
    # path metric and best input bit for each state at each time.
    for k in range(n_in_bits - 1, -1, -1):
      self._update_path_metrics(
      code_bit_llrs[self.n_out * k : self.n_out * (k + 1)],path_metrics, best_bit[k])

    # Decode by starting in state 0 at time 0 and tracing path
    # corresponding to best input bits.
    info_bits_hat, enc_state = -np.ones(n_info_bits, dtype=int), 0
    for k in range(n_info_bits):
      info_bits_hat[k] = best_bit[k][enc_state]
      enc_state = self.next_state[info_bits_hat[k]][enc_state]

    return info_bits_hat

  def _update_alpha(self,out_bit_llrs,pre_in_bit_llr,alpha_val,alpha_val_next,max_log):

    gamma_val = self._branch_metrics(out_bit_llrs, pre_in_bit_llr)

    for enc_state in self.state_space:
      alpha_val_next[self.next_state[0][enc_state]] = self.maxstar(alpha_val_next[self.next_state[0][enc_state]],alpha_val[enc_state] + gamma_val[0][enc_state],max_log)
      alpha_val_next[self.next_state[1][enc_state]] = self.maxstar(alpha_val_next[self.next_state[1][enc_state]],alpha_val[enc_state] + gamma_val[1][enc_state],max_log)

    return


# In[110]:


class conv_code(conv_code):

  def _update_beta_tail(self, out_bit_llrs, beta_val, max_log):

    gamma_val = self._branch_metrics(out_bit_llrs, 0)

    bvn = beta_val[:]
    for enc_state in self.state_space:
      beta_val[enc_state] = self.maxstar(gamma_val[0][enc_state] + bvn[self.next_state[0][enc_state]],gamma_val[1][enc_state] + bvn[self.next_state[1][enc_state]],max_log)
    return

  def _update_beta(self,out_bit_llrs,pre_in_bit_llr,alpha_val,beta_val,max_log):

    gamma_val = self._branch_metrics(out_bit_llrs, pre_in_bit_llr)

    met0 = -conv_code.INF
    met1 = -conv_code.INF
    bvn = beta_val[:]
    for enc_state in self.state_space:
      beta_val[enc_state] = self.maxstar(gamma_val[0][enc_state] + bvn[self.next_state[0][enc_state]],gamma_val[1][enc_state] + bvn[self.next_state[1][enc_state]],max_log)
      met0 = self.maxstar(alpha_val[enc_state] + gamma_val[0][enc_state]+ bvn[self.next_state[0][enc_state]],met0,max_log)
      met1 = self.maxstar(alpha_val[enc_state] + gamma_val[1][enc_state]+ bvn[self.next_state[1][enc_state]],met1,max_log)
    return met0 - met1


# In[111]:


class conv_code(conv_code):
  def decode_bcjr(self,code_bit_llrs,pre_info_bit_llrs=None,max_log=True):

    code_bit_llrs = np.asarray(code_bit_llrs).ravel()
    n_in_bits = int(code_bit_llrs.size / self.n_out)
    n_info_bits = n_in_bits - self.mem_len

    if pre_info_bit_llrs is None:
      pre_info_bit_llrs = np.zeros(n_info_bits)
    else:
      pre_info_bit_llrs = np.asarray(pre_info_bit_llrs).ravel()

    # FORWARD PASS: Recursively compute alpha values for all states at
    # all times from 1 to n_info_bits - 1, working forward from time 0.
    alpha = [[(0 if s == 0 and k == 0 else -conv_code.INF)for s in self.state_space] for k in range(n_info_bits)]
    for k in range(n_info_bits - 1):
      out_bit_llrs = code_bit_llrs[self.n_out * k : self.n_out * (k + 1)]
      self._update_alpha(out_bit_llrs, pre_info_bit_llrs[k],alpha[k], alpha[k + 1], max_log)

    # BACKWARD PASS (TAIL): Recursively compute beta values for all
    # states at time n_info_bits, working backward from time n_in_bits.
    beta = [(0 if s == 0 else -conv_code.INF) for s in self.state_space]
    for k in range(n_in_bits - 1, n_info_bits - 1, -1):
      out_bit_llrs = code_bit_llrs[self.n_out * k : self.n_out * (k + 1)]
      self._update_beta_tail(out_bit_llrs, beta, max_log)

    # BACKWARD PASS: Recursively compute beta values for all states at
    # each time k from 0 to n_info_bits - 1, working backward from time
    # n_info_bits, and also obtaining the post-decoding LLR for the info
    # bit at each time.
    post_info_bit_llrs = np.zeros_like(pre_info_bit_llrs)
    for k in range(n_info_bits - 1, - 1, -1):
      out_bit_llrs = code_bit_llrs[self.n_out * k : self.n_out * (k + 1)]
      post_info_bit_llrs[k] = self._update_beta(out_bit_llrs, pre_info_bit_llrs[k],alpha[k], beta, max_log)

    return post_info_bit_llrs


# In[112]:


class coding():

  def __init__(self,K):
    #information length
    
    self.output_all_iterations=False #:default:false 
    self.K=K
    #self.N=1024##may use(under construnction)
    self.R=1/2#self.K/self.N
    self.N=int(self.K/self.R)
    print("R")
    print(self.R)
    self.back_poly=13
    self.parity_polys=[11]
    self.max_itr=18
    
    if self.R<1/2:
      self.mod=2
    else:
      self.mod=math.floor(2/(1/self.R-1))
    
    #punctur index
    if self.R!=1/3:
      self.ind_top,self.ind_bot=self.make_puncture_indeces(self.N,self.K,self.R)
    
    #print(self.ind_top)
    #print(self.ind_bot)

    #to write txt file
    self.filename="turbo_code_{}_{}".format(self.K,math.floor(self.R*100)/100) #rateは少数第二桁まで表示

    # Encoder and decoder for constituent RSC code
    self.rsc = conv_code(self.back_poly, [self.back_poly] + self.parity_polys)

    # Number of output bits per input bit and number of tail bits
    # per input block for the turbo code

    self.n_out = self.rsc.n_out + (self.rsc.n_out - 1)

    self.n_tail_bits = self.rsc.n_out * self.rsc.mem_len * 2

    # Turbo interleaver and deinterleaver
    self.turbo_int, self.turbo_deint =  self.interleave()

    return

  def interleave(self):
    #make s-random odd-even interleaver sequence
    turbo_int=self.make_interleaver_sequence()
    turbo_deint=np.argsort(turbo_int)
    return turbo_int,turbo_deint


# In[113]:


class coding(coding):

  def make_puncture_indeces(self,N,K,R):
    
    parity_num=N-K #parityビットの数
    
    if R>=1/2:

      #パリティビットの個数分のインデックスを作成
      ind_top=np.random.choice(K//self.mod,size=math.floor(parity_num/2),replace=False)
      ind_bot=np.random.choice(K//self.mod,size=math.ceil(parity_num/2),replace=False)

      #paritybitのあるインデックスの配列
      ind_top=self.mod*ind_top
      ind_bot=self.mod*ind_bot+1
      
      res_top = np.ones(K, dtype=bool)
      res_top[ind_top]=False
      res_bot = np.ones(K, dtype=bool)
      res_bot[ind_bot]=False 
      
    else: #if 1/3<rate<1/2

      parity_num=parity_num-K 
      if parity_num<0:
        print("error")
              
      res_top = np.ones(K, dtype=bool)
      res_top[0::2]=False
      res_bot = np.ones(K, dtype=bool)
      res_bot[1::2]=False
      
      ind_top_rem=np.random.choice(K//self.mod,size=math.floor(parity_num/2),replace=False)
      ind_bot_rem=np.random.choice(K//self.mod,size=math.ceil(parity_num/2),replace=False)

      #paritybitのあるインデックスの配列
      ind_top_rem=self.mod*ind_top_rem+1
      ind_bot_rem=self.mod*ind_bot_rem
      
      #print(ind_top_rem)
      #print(ind_bot_rem)
      
      res_top[ind_top_rem]=False
      res_bot[ind_bot_rem]=False
    
    
    if (len(res_top[res_top==False])+len(res_bot[res_bot==False]))!=(N-K):
      print("puncture error")
          
    return res_top,res_bot


# In[114]:


class coding(coding):

  def make_interleaver_sequence(self):
    s=math.floor(math.sqrt(self.K))-10
    print(s)
    #step 1 generate random sequence
    vector=np.arange(self.K,dtype='int')
    np.random.shuffle(vector)

    itr=True
    count=0
    while itr:
      #intialize for each iteration
      heap=np.zeros(self.K,dtype='int')
      position=np.arange(self.K,dtype='int')

      #step2 set first vector to heap
      heap[0]=vector[0]
      position=np.delete(position,0)

      #step3 bubble sort 
      #set to ith heap
      for i in range(1,self.K):
        #serch jth valid position
        for pos,j in enumerate(position):
          # confirm valid or not
          for k in range(1,s+1):
            if i-k>=0 and (abs(heap[i-k]-vector[j])+abs(i-k-j))<=s or (vector[j]%self.mod)!=(i%self.mod):
              '''
              i-k>=0 : for the part i<s 
              (abs(heap[i-k]-vector[j]))<=s : srandom interleaver
              vector[j]//mod!=i//mod : mod M interleaver(such as odd-even)
              '''
              #vector[j] is invalid and next vector[j+1]
              break

          #vector[j] is valid and set to heap[i]
          else:
            heap[i]=vector[j]
            position=np.delete(position,pos)
            break

        #if dont exit num at heap[i]
        else:
          #set invalid sequence to the top and next iteration
          tmp=vector[position]
          np.random.shuffle(tmp)
          vector[0:self.K-i]=tmp
          vector[self.K-i:self.K]=heap[0:i]
          break

      #if all the heap num is valid, end iteration
      else:
          itr=False
      
      #print(heap)
      #print(vector)
      print("\r","itr",count,end="")
      count+=1
      
    return heap


# In[115]:


class encoding(coding):

  def __init__(self,K):
    super().__init__(K)

  def generate_information(self):
    information=np.random.randint(0,2,self.K)
    return information

  def encoding(self, information):
    self.rsc = conv_code(self.back_poly, [self.back_poly] + self.parity_polys)
    #print("info is")
    #print(len(information))
    # Get code bits from each encoder.
    ctop = self.rsc.encode(information)
    cbot = self.rsc.encode(information[self.turbo_int])
    #print("cwd is")
    #print(len(ctop))
    #print(len(cbot))
    
    #puncturing 
    #if self.R!=1/3:
      #pick up parity
      #ptop=ctop[1:2*self.K:2]
      #pbot=cbot[1:2*self.K:2]
  
      #insert 0
      #ptop[self.ind_top]=0
      #pbot[self.ind_bot]=0
      
      #make codeword
      #ctop[1:2*self.K:2]=ptop
      #cbot[1:2*self.K:2]=pbot
    
    # Assemble code bits from both encoders.
    codeword, pos = -np.ones(self.n_out * self.K + self.n_tail_bits, dtype=int), 0

    for k in range(self.K):
      codeword[pos : pos + self.rsc.n_out] = ctop[self.rsc.n_out * k : self.rsc.n_out * (k + 1)]
      pos += self.rsc.n_out
      codeword[pos : pos + self.rsc.n_out - 1] = cbot[self.rsc.n_out * k + 1 : self.rsc.n_out * (k + 1)]
      pos += self.rsc.n_out - 1
      
    codeword[pos : pos + self.rsc.n_out * self.rsc.mem_len] = ctop[self.rsc.n_out * self.K :]
    codeword[pos + self.rsc.n_out * self.rsc.mem_len :] = cbot[self.rsc.n_out * self.K :]

    return codeword


# In[116]:


class encoding(encoding):
  def turbo_encode(self):
    information=self.generate_information()
    codeword=self.encoding(information)
    
    return information,codeword


# In[117]:


class decoding(coding):

  def __init__(self,K):
    super().__init__(K)
  
  def decoding(self, Lc):

    # Systematic bit LLRs for each decoder
    lambda_s = Lc[0 : self.n_out * self.K : self.n_out]
    in_lambda_s = lambda_s[self.turbo_int]

    # Code bit LLRs for each decoder
    ctop_llrs = np.zeros(self.rsc.n_out * (self.K + self.rsc.mem_len))
    cbot_llrs = np.zeros(self.rsc.n_out * (self.K + self.rsc.mem_len))
    pos = 0

    for k in range(self.K):
      num = self.rsc.n_out * k
      ctop_llrs[num] = lambda_s[k]
      cbot_llrs[num] = in_lambda_s[k]
      pos += 1
      ctop_llrs[num + 1 : num + self.rsc.n_out] = Lc[pos : pos + self.rsc.n_out - 1]
      pos += self.rsc.n_out - 1
      cbot_llrs[num + 1 : num + self.rsc.n_out] = Lc[pos : pos + self.rsc.n_out - 1]
      pos += self.rsc.n_out - 1
      
    ctop_llrs[self.rsc.n_out * self.K :] = Lc[pos : pos + self.rsc.n_out * self.rsc.mem_len]
    cbot_llrs[self.rsc.n_out * self.K :] = Lc[pos + self.rsc.n_out * self.rsc.mem_len :]

    #puncturing 
    if self.R!=1/3:
      #pick up parity
      ptop_llrs=ctop_llrs[1:2*self.K:2]
      pbot_llrs=cbot_llrs[1:2*self.K:2]
  
      #insert 0
      ptop_llrs[self.ind_top]=0
      pbot_llrs[self.ind_bot]=0
      
      #make codeword
      ctop_llrs[1:2*self.K:2]=ptop_llrs
      cbot_llrs[1:2*self.K:2]=pbot_llrs
    

    # Main loop for turbo iterations
    if self.output_all_iterations:#output 2D EST_information
      EST_information=np.zeros((self.K,self.max_itr))
    
    
    lambda_e, in_lambda_e = np.zeros(self.K), np.zeros(self.K)
    for i in range(self.max_itr):
      res = self.rsc.decode_bcjr(ctop_llrs, in_lambda_e[self.turbo_deint])
      lambda_e = res- in_lambda_e[self.turbo_deint] - lambda_s
      in_res = self.rsc.decode_bcjr(cbot_llrs, lambda_e[self.turbo_int])
      in_lambda_e = in_res - lambda_e[self.turbo_int] - in_lambda_s
      
      if self.output_all_iterations:#output 2D EST_information
        res = in_res[self.turbo_deint]
        EST_information[:,i] = (res < 0).astype(int)

    # Final post-decoding LLRs and hard decisions
    if self.output_all_iterations==False:
      res = in_res[self.turbo_deint]
      EST_information = (res < 0).astype(int)
      
    #print(EST_information.shape)

    return EST_information


# In[118]:


class decoding(decoding):
  def turbo_decode(self,Lc):
    EST_information=self.decoding(Lc)
    return EST_information


# In[119]:


class turbo_code(encoding,decoding):

  def __init__(self,K):
    super().__init__(K)
    
  def main_func(self,EbNodB): 
    information,codeword=self.turbo_encode()
    Lc=-1*ch.generate_LLR(codeword,EbNodB)  ###LLR reverse
    EST_information=self.turbo_decode(Lc) 

    return information,EST_information

# In[120]:

if __name__=="__main__":
  tc=turbo_code(500)
  #print(tc.turbo_int)
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

      #main calcuration
      information,EST_information=tc.main_func(EbNodB)
      print(EST_information.shape)
      #EST_information=EST_information[:,EST_information.shape[1]-1]
      #information,EST_information=tc.main_func(EbNodB)
      
      #calculate block error rate
      if np.any(information!=EST_information):
          count_err+=1
      count_all+=1

      #calculate bit error rate 
      count_berr+=np.sum(information!=EST_information)
      #count_berr_mat+=(information!=EST_information)
      #print(count_berr_mat)
      count_ball+=len(information)

      print("\r","count_all=",count_all,",count_err=",count_err,"count_ball=",count_ball,"count_berr=",count_berr,end="")

    return count_err,count_all,count_berr,count_ball

  EbNodB_range=np.array([-1])
  for i in EbNodB_range:
    print(i)
    _,_,a,b=output(i)
    print(a/b)
    

