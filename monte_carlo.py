#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import ray
import pickle
from turbo_code import turbo_code
from AWGN import _AWGN


# In[4]:


ray.init()


# In[ ]:


@ray.remote
def output(dumped,EbNodB):
        '''
        #あるSNRで計算結果を出力する関数を作成
        #cd.main_func must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'
        '''

        #de-seriallize file
        cd=pickle.loads(dumped)
        #seed値の設定
        np.random.seed()

        #prepare some constants
        #MAX_ERR=1
        MAX_BITALL=10**6
        count_bitall=0
        count_biterr=0
        count_all=0
        count_err=0
        

        while count_bitall<MAX_BITALL:
        #print("\r"+str(count_err),end="")
            information,EST_information=cd.main_func(EbNodB)
            
            #calculate block error rate
            if np.any(information!=EST_information):
                count_err+=1
            count_all+=1

            #calculate bit error rate 
            count_biterr+=np.sum(information!=EST_information)
            count_bitall+=len(information)

        return count_err,count_all,count_biterr,count_bitall


# In[11]:


class MC():
    def __init__(self):
        self.TX_antenna=1
        self.RX_antenna=1
        self.MAX_ERR=8
        self.EbNodB_start=-5
        self.EbNodB_end=2
        self.EbNodB_range=np.arange(self.EbNodB_start,self.EbNodB_end,1) #0.5dBごとに測定
        self.max_itr=18

    #特定のNに関する出力
    def monte_carlo_get_ids(self,dumped):
        '''
        input:main_func
        -----------
        dumped:seriallized file 
        main_func: must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'
        -----------
        output:result_ids(2Darray x:SNR_number y:MAX_ERR)

        '''

        print("from"+str(self.EbNodB_start)+"to"+str(self.EbNodB_end))
        
        result_ids=[[] for i in range(len(self.EbNodB_range))]

        for i,EbNodB in enumerate(self.EbNodB_range):
            
            for j in range(self.MAX_ERR):
                #multiprocess    
                result_ids[i].append(output.remote(dumped,EbNodB))  # 並列演算
                #resultは長さ1のリストの中にBLER,BERの2つのarrayのtupleが入った配列
        
        return result_ids
    
    def monte_carlo_calc(self,result_ids_array,N_list):

        #prepare constant
        tmp_num=self.MAX_ERR
        tmp_ids=[]

        #Nのリストに対して実行する
        for i,N in enumerate(N_list):
            #特定のNに対して実行する
            #特定のNのBER,BLER系列
            BLER=np.zeros(len(self.EbNodB_range))
            BER=np.zeros(len(self.EbNodB_range))

            for j,EbNodB in enumerate(self.EbNodB_range):#i=certain SNR
                
                #特定のSNRごとに実行する
                while sum(np.isin(result_ids_array[i][j], tmp_ids)) == len(result_ids_array[i][j]):#j番目のNの、i番目のSNRの計算が終わったら実行
                    finished_ids, running_ids = ray.wait(result_ids_array[i], num_returns=tmp_num, timeout=None)
                    tmp_num+=1
                    tmp_ids=finished_ids

                result=ray.get(result_ids_array[i][j])
                #resultには同じSNRのリストが入る
                count_err=0
                count_all=0
                count_biterr=0
                count_bitall=0
                
                for k in range(self.MAX_ERR):
                    tmp1,tmp2,tmp3,tmp4=result[k]
                    count_err+=tmp1
                    count_all+=tmp2
                    count_biterr+=tmp3
                    count_bitall+=tmp4

                BLER[j]=count_err/count_all
                BER[j]=count_biterr/count_bitall

                #if count_biterr/count_bitall<10**-5:
                    #print("finish")
                    #break

                print("\r"+"EbNodB="+str(EbNodB)+",BLER="+str(BLER[j])+",BER="+str(BER[j]),end="")
            
            #特定のNについて終わったら出力
            st=savetxt(N)
            st.savetxt(BLER,BER)



# In[ ]:


#毎回書き換える関数
class savetxt(turbo_code,_AWGN,MC):

  def __init__(self,K):
    super().__init__(K)   

  def savetxt(self,BLER,BER):

    with open(self.filename,'w') as f:

        print("#N="+str(self.K),file=f)
        print("#TX_antenna="+str(self.TX_antenna),file=f)
        print("#RX_antenna="+str(self.RX_antenna),file=f)
        print("#modulation_symbol="+str(self.M),file=f)
        print("#MAX_BLER="+str(self.MAX_ERR),file=f)
        print("#iteration number="+str(self.max_itr),file=f)
        print("#EsNodB,BLER,BER",file=f) 
        for i in range(len(self.EbNodB_range)):
            print(str(self.EbNodB_range[i]),str(BLER[i]),str(BER[i]),file=f)


# In[ ]:

if __name__=="__main__":
    mc=MC()

    N_list=[512,1024,2048,4096]
    result_ids_array=[]
    print(mc.EbNodB_range)
    for i,N in enumerate(N_list):
        cd=turbo_code(N)
        dumped=pickle.dumps(cd)
        print("N=",N)
        result_ids_array.append(mc.monte_carlo_get_ids(dumped))
    
    mc.monte_carlo_calc(result_ids_array,N_list)