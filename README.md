panda, a framework co-processing spmd computation on gpus and cpus
=================================================================
		li7hui@gmail.com
		baoding yunmu co.ltd
		2017.11.09

steps to run sample: 
	1)make in gpumapreduce
	2)mpirun -host node1,node2 -np 2 ./panda_word_count input.txt
