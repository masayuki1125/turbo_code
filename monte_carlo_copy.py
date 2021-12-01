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
        #cd.main_func must input 'EbNodB' and output 1D or 2D 'codeword' and 'EST_codeword'
        '''

        #de-seriallize file
        cd=pickle.loads(dumped)
        #seed値の設定
        np.random.seed()

        #prepare some constants
        MAX_ERR=1
        MAX_ALL=10**5
        MAX_BITALL=10**6
        count_bitall=np.zeros((1))
        count_biterr=np.zeros((1))
        count_all=np.zeros((1))
        count_err=np.zeros((1))
        

        while count_bitall[len(count_err)-1]<MAX_BITALL: #count_all<MAX_ALL エラーの個数を指定するか試行回数を指定するかを選ぶ #max itr番目のデータのエラーカウントをチェック
       
            information,EST_information=cd.main_func(EbNodB) #matrixでのESTinformationにも対応
            
            if EST_information.ndim==1: #change EST_information to 2D 
                EST_information=EST_information[:,np.newaxis]
                
            data_len=EST_information.shape[1]
            
            #count変数のベクトルを変更(最初のイテレーション時に実行)
            if len(count_all)!=data_len:
                count_bitall=np.zeros((data_len))
                count_biterr=np.zeros((data_len))
                count_all=np.zeros((data_len))
                count_err=np.zeros((data_len))
            
            for i in range(data_len): #EST_information1列ごとに足し算
                #calculate block error rate
                if np.any(information!=EST_information[:,i]):
                    count_err[i]+=1
                count_all[i]+=1

                #calculate bit error rate 
                count_biterr[i]+=np.sum(information!=EST_information[:,i])
                count_bitall[i]+=len(information)

        return count_err,count_all,count_biterr,count_bitall


# In[11]:


class MC():
    def __init__(self):
        self.TX_antenna=1
        self.RX_antenna=1
        self.parallel=100
        self.EbNodB_start=-5
        self.EbNodB_end=1
        self.EbNodB_range=np.arange(self.EbNodB_start,self.EbNodB_end,0.5) #0.5dBごとに測定

    #特定のNに関する出力
    def monte_carlo_get_ids(self,dumped):
        '''
        input:main_func
        -----------
        dumped:seriallized file 
        main_func: must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'
        -----------
        output:result_ids(2Darray x:SNR_number y:parallel)

        '''

        print("from"+str(self.EbNodB_start)+"to"+str(self.EbNodB_end))
        
        result_ids=[[] for i in range(len(self.EbNodB_range))]

        for i,EbNodB in enumerate(self.EbNodB_range):
            
            for j in range(self.parallel):
                #multiprocess    
                result_ids[i].append(output.remote(dumped,EbNodB))  # 並列演算
                #resultは長さ1のリストの中にBLER,BERの2つのarrayのtupleが入った配列
        
        return result_ids
    
    def monte_carlo_calc(self,result_ids_array,N_list):

        #prepare constant
        tmp_num=self.parallel
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
                
                for k in range(self.parallel):
                    tmp1,tmp2,tmp3,tmp4=result[k]
                    count_err+=tmp1
                    count_all+=tmp2
                    count_biterr+=tmp3
                    count_bitall+=tmp4
                    
                #count_allの要素数が１でなかったときに最初の反復のときのみ実行
                if len(count_all)!=1 and j==0:
                    data_len=len(count_all)
                    #BER,BLERを二次元に拡張
                    BLER=np.zeros((len(self.EbNodB_range),data_len))
                    BER=np.zeros((len(self.EbNodB_range),data_len))

                BLER[j]=count_err/count_all
                BER[j]=count_biterr/count_bitall

                #if count_biterr/count_bitall<10**-4:
                    #print("finish")
                    #break

                print("\r"+"EbNodB="+str(EbNodB)+",BLER="+str(BLER[j])+",BER="+str(BER[j]),end="")
            
            #特定のNについて終わったら出力
            print(BLER)
            st=savetxt(N)
            st.savetxt(BLER,BER)



# In[ ]:


#毎回書き換える関数
class savetxt():

  def __init__(self,N):
    self.tc=turbo_code(N)
    self.mc=MC()
    self.ch=_AWGN()

  def savetxt(self,BLER,BER):
      
    #BLERが二次元のときに対応
    
    if BLER.ndim==1: #change EST_information to 2D 
        BLER=BLER[:,np.newaxis]
        BER=BER[:,np.newaxis]
        
        
    data_len=BLER.shape[1]
      
      
    for j in range(data_len):
      new_filename=self.tc.filename
      if BLER.shape[1]!=1:
        new_filename=self.tc.filename+"_"+str(j+1) #ファイル名にイテレーション回数を記入

      with open(new_filename,'w') as f:

        print("#N="+str(self.tc.K),file=f)
        print("#TX_antenna="+str(self.ch.TX_antenna),file=f)
        print("#RX_antenna="+str(self.ch.RX_antenna),file=f)
        print("#modulation_symbol="+str(self.ch.M),file=f)
        print("#max_itr="+str(self.tc.max_itr),file=f)
        print("#parallel="+str(self.mc.parallel),file=f)
        print("#EsNodB,BLER,BER",file=f) 
        
        for i in range(len(self.mc.EbNodB_range)):
            print(str(self.mc.EbNodB_range[i]),str(BLER[i,j]),str(BER[i,j]),file=f)


# In[ ]:
if __name__=="__main__":
    mc=MC()

    N_list=[512]
    result_ids_array=[]
    print(mc.EbNodB_range)
    for i,N in enumerate(N_list):
        cd=turbo_code(N)
        dumped=pickle.dumps(cd)
        print("N=",N)
        result_ids_array.append(mc.monte_carlo_get_ids(dumped))

    mc.monte_carlo_calc(result_ids_array,N_list)
