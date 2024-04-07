Failed Code with latest CUDA toolkit
Panda, a heterogeneous MapReduce framework on GPUs and CPUs cluster
=================================================================
		huili@ruijie.com.cn
		Ruijie network co.ltd
		2017.11.09
  		li7hui@gmail.com
		First Prize in MCM 2002
		WuShi Prize in PKU 2008
		IEEE Cluster Paper in 2013 in Indianapolis

Paper: docs/Co-processing SPMD computation on CPUs and GPUs cluster.pdf

Support apps:
1) word count
2) terasort

-----------------------------------------------------------------
The code is tested using Ubuntu 22.04+openmpi-5.0.0+cuda-11.5.0 
Steps to compile openmpi-5.0.0
1) wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.0.tar.gz
2) ./configure --prefix=/opt/openmpi/
3) make -j 2
4) make install
-----------------------------------------------------------------
steps to run word count:<br>
    1)cd gpumapreduce<br>
	2)make wordcount<br>
	3)cd bin<br>
	4)mpirun -host node1,node2 -np 2 ./wordcount input2.txt<br>
	5)cat ./OUTPUT0<br>
<br>
steps to run terasort:<br>
	1)cd gpumapreduce<br>
	2)make terasort<br>
	3)cd bin<br>
	4)mpirun -host node1,node2 -np 2 ./teragen 10M /tmp/terasort_in<br>
	5)mpirun -host node1,node2 -np 2 ./terasort ./tmp/terasort_in ./tmp/terasort_out<br>

------------------------------------------------------------------
The code structure:<br>

gpumapreduce/<br>
<br>
