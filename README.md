panda, a framework co-processing spmd computation on gpus and cpus
=================================================================
		li7hui@gmail.com
		ruijie network co.ltd
		2017.11.09

Title: docs/Co-processing SPMD computation on CPUs and GPUs cluster.pdf

Support apps:
1) word count
2) cmeans

-----------------------------------------------------------------
steps to run sample:<br>
        1)cd gpumapreduce<br>
	2)make<br>
	3)mpirun -host node1,node2 -np 2 ./panda_word_count input.txt<br>
