# Meta Information of the Code
(1) ICPPAppendix is the full proof for ICPP 2019. 

>Wang, Haozhao, Song Guo, and Ruixuan Li. "Osp: Overlapping computation and communication in parameter server for fast machine learning." Proceedings of the 48th International Conference on Parallel Processing. 2019.

(2) The four file folders are the source codes for OSP and its comparison models.

# Run the program
**(1)Dependency**

The program is based on PyTorch 1 and Python 3.

**(2) Start orders**

*(a) Start Server*

>python param_server.py
>>--ps-ip
>>
>>--ps-port
>>
>>--this-rank=0
>>
>>--workers-num
>>
>>--model
>>
>>--epochs
>>
>>--train-bsz

*(b) Start a learner*

> python learner.py 
>>      --ps-ip 
>>      
>>--ps-port
>>
>>--this-rank
>>
>>--workers-num 
>>
>>--model
>>
>>--epochs
>>
>>--train-bsz 
>>
>>--stale-threshold

           

