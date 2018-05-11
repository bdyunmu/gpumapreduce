Panda, a framework co-processing SPMD computation on GPUs and CPUs
=================================================================
		huili@ruijie.com.cn
		ruijie network co.ltd
		2017.11.09

Title: docs/Co-processing SPMD computation on CPUs and GPUs cluster.pdf

Support apps:
1) word count
2) cmeans

-----------------------------------------------------------------
steps to run sample:<br>
        1)cd gpumapreduce<br>
	2)make wordcount<br>
	3)cd bin<br>
	4)mpirun -host node1,node2 -np 2 ./wordcount input.txt<br>
