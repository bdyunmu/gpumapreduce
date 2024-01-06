Panda, a heterogeneous MapReduce framework on GPUs and CPUs cluster
=================================================================
		huili@ruijie.com.cn
		Ruijie network co.ltd
		2017.11.09
		lihui@zhongjuai.com
		Zhongjuai co.ltd
		2023.10.29

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
The code structure:

gpumapreduce/
├── apps
│   ├── terasort
│   │   ├── Makefile
│   │   ├── Makefile.old
│   │   ├── Makefile.old.2
│   │   ├── parameters.sh
│   │   ├── Random16.cpp
│   │   ├── Random16.h
│   │   ├── Random16.java
│   │   ├── run.sh
│   │   ├── teragen
│   │   ├── teragen_api.cu
│   │   ├── teragen_main.cpp
│   │   ├── TeraGen.scala
│   │   ├── TeraInputFormat.cpp
│   │   ├── TeraInputFormat.h
│   │   ├── TeraInputFormat.scala
│   │   ├── TeraOutputFormat.scala
│   │   ├── terasort_api.cu
│   │   ├── terasort.h
│   │   ├── terasort_main.cpp
│   │   ├── TeraSortPartitioner.cpp
│   │   ├── TeraSortPartitioner.h
│   │   ├── TeraSortPartitioner.scala
│   │   ├── TeraSort.scala
│   │   ├── teravalidate.cpp
│   │   ├── TeraValidate.scala
│   │   ├── tsoutputformat.cpp
│   │   ├── tsoutputformat.h
│   │   ├── Unsigned16.cpp
│   │   ├── Unsigned16.h
│   │   └── Unsigned16.java
│   └── wordcount
│       ├── Makefile
│       ├── wc_api.cu
│       ├── wc_main.cpp
│       ├── wcoutputformat.cpp
│       └── wcoutputformat.h
├── bin
│   ├── input2.txt
│   ├── OUTPUT0
│   ├── terasort_in
│   │   ├── INPUT0
│   │   └── INPUT1
│   └── wordcount
├── docs
│   └── Co-processing SPMD computation on CPUs and GPUs cluster.pdf
├── include
│   ├── cudacpp
│   │   ├── Event.h
│   │   ├── myString.h
│   │   ├── Runtime.h
│   │   └── Stream.h
│   ├── oscpp
│   │   ├── AsyncFileReader.h
│   │   ├── AsyncIORequest.h
│   │   ├── Condition.h
│   │   ├── GenericAsyncIORequest.h
│   │   ├── Mutex.h
│   │   ├── Runnable.h
│   │   ├── Thread.h
│   │   └── Timer.h
│   ├── panda
│   │   ├── Chunk.h
│   │   ├── Combiner.h
│   │   ├── DataChunk.h
│   │   ├── EmitConfiguration.h
│   │   ├── FileChunk.h
│   │   ├── FileOutput.h
│   │   ├── KeyValueChunk.h
│   │   ├── Mapper.h
│   │   ├── MapReduceJob.h
│   │   ├── Message.h
│   │   ├── OutputAsInput.h
│   │   ├── Output.h
│   │   ├── PandaCPUConfig.h
│   │   ├── PandaGPUConfig.h
│   │   ├── PandaMapReduceJob.h
│   │   ├── PandaMessage.h
│   │   ├── PandaMessageIORequest.h
│   │   ├── PandaMessagePackage.h
│   │   ├── PandaMPIMessage.h
│   │   ├── Partitioner.h
│   │   └── Reducer.h
│   ├── PandaAPI.h
│   └── Panda.h
├── LICENSE
├── Makefile
├── makefile.in
├── README.md
├── splitmpisendasanotherthread
└── src
    ├── cudacpp
    │   └── Stream.cpp
    ├── inputformat
    │   ├── Chunk.cpp
    │   ├── DataChunk.cpp
    │   ├── FileChunk.cpp
    │   └── KeyValueChunk.cpp
    ├── message
    │   ├── Messsage.cpp
    │   ├── PandaMessage.cpp
    │   ├── PandaMessageIORequest.cpp
    │   └── PandaMPIMessage.cpp
    ├── oscpp
    │   ├── AsyncFileReader.cpp
    │   ├── AsyncIORequest.cpp
    │   ├── Condition.cpp
    │   ├── Mutex.cpp
    │   ├── Runnable.cpp
    │   ├── Thread.cpp
    │   └── Timer.cpp
    ├── outputformat
    │   └── Output.cpp
    ├── pandajob
    │   ├── Compare.cpp
    │   ├── MapReduceJob.cpp
    │   ├── PandaMapReduceJob.cpp
    │   └── Partitioner.cpp
    └── runtime
        ├── PandaLib.cu
        ├── PandaSched.cu
        ├── PandaSort.cu
        └── PandaUtils.cu
