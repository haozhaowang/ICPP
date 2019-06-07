# ICPP
(1) ICPPAppendix is the full proof for ICPP 2019, "OSP: Overlapping Computation and Communication in Parameter Server".
(2) The other four file folders are the source code for OSP and its comparison models.

# Execute
The program is based on PyTorch 1 and Python 3.
The order of starting OSP is:
    (a) python param_server.py 
           --ps-ip
           --ps-port
           --this-rank=0
           --workers-num
           --model
           --epochs
           --train-bsz
    (b) python learner.py 
           --ps-ip
           --ps-port
           --this-rank
           --workers-num
           --model
           --epochs
           --train-bsz
           --stale-threshold

           

