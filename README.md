Panda, a heterogeneous MapReduce framework on GPUs and CPUs cluster
=================================================================
		huili@ruijie.com.cn
		Ruijie network co.ltd
		2017.11.09

Paper: docs/Co-processing SPMD computation on CPUs and GPUs cluster.pdf

Support apps:
1) word count
2) terasort

-----------------------------------------------------------------
steps to run word count:<br>
        1)cd gpumapreduce<br>
	2)make wordcount<br>
	3)cd bin<br>
	4)mpirun -host node1,node2 -np 2 ./wordcount input.txt<br>
	5)cat ./OUTPUT0
<br>
steps to run terasort:<br>
	1)cd gpumapreduce<br>
	2)make terasort<br>
	3)cd bin<br>
	4)mpirun -host node1,node2 -np 2 ./teragen 10M /tmp/terasort_in<br>
	5)mpirun -host node1,node2 -np 2 ./terasort ./tmp/terasort_in ./tmp/terasort_out<br>
