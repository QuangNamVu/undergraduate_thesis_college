# grep -B 2 -A 1 "(global_step 4000) finished\," log_nz  |grep -E accu\|saved |grep -A 1 "accuracy_: 0\.[5-9]"> summary_nz
# grep -B 2 -A 1 "(global_step 4000) finished\," log  |grep -E accu\|saved |grep -A 1 "accuracy_: 0\.[5-9]"> summary_T_d 
grep -B 2 -A 1 "(global_step 4000) finished\," log.txt  |grep -E accu\|saved |grep -A 1 "accuracy_: 0\.[5-9]"> summary_alpha
wc summary_*
