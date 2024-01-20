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
├── apps<br>
│   ├── terasort<br>
│   │   ├── Makefile<br>
│   │   ├── Makefile.old<br>
│   │   ├── Makefile.old.2<br>
│   │   ├── parameters.sh<br>
│   │   ├── Random16.cpp<br>
│   │   ├── Random16.h<br>
│   │   ├── Random16.java<br>
│   │   ├── run.sh<br>
│   │   ├── teragen<br>
│   │   ├── teragen_api.cu<br>
│   │   ├── teragen_main.cpp<br>
│   │   ├── TeraGen.scala<br>
│   │   ├── TeraInputFormat.cpp<br>
│   │   ├── TeraInputFormat.h<br>
│   │   ├── TeraInputFormat.scala<br>
│   │   ├── TeraOutputFormat.scala<br>
│   │   ├── terasort_api.cu<br>
│   │   ├── terasort.h<br>
│   │   ├── terasort_main.cpp<br>
│   │   ├── TeraSortPartitioner.cpp<br>
│   │   ├── TeraSortPartitioner.h<br>
│   │   ├── TeraSortPartitioner.scala<br>
│   │   ├── TeraSort.scala<br>
│   │   ├── teravalidate.cpp<br>
│   │   ├── TeraValidate.scala<br>
│   │   ├── tsoutputformat.cpp<br>
│   │   ├── tsoutputformat.h<br>
│   │   ├── Unsigned16.cpp<br>
│   │   ├── Unsigned16.h<br>
│   │   └── Unsigned16.java<br>
│   └── wordcount<br>
│       ├── Makefile<br>
│       ├── wc_api.cu<br>
│       ├── wc_main.cpp<br>
│       ├── wcoutputformat.cpp<br>
│       └── wcoutputformat.h<br>
├── bin<br>
│   ├── input2.txt<br>
│   ├── OUTPUT0<br>
│   ├── terasort_in<br>
│   │   ├── INPUT0<br>
│   │   └── INPUT1<br>
│   └── wordcount<br>
├── docs<br>
│   └── Co-processing SPMD computation on CPUs and GPUs cluster.pdf<br>
├── include<br>
│   ├── cudacpp<br>
│   │   ├── Event.h<br>
│   │   ├── myString.h<br>
│   │   ├── Runtime.h<br>
│   │   └── Stream.h<br>
│   ├── oscpp<br>
│   │   ├── AsyncFileReader.h<br>
│   │   ├── AsyncIORequest.h<br>
│   │   ├── Condition.h<br>
│   │   ├── GenericAsyncIORequest.h<br>
│   │   ├── Mutex.h<br>
│   │   ├── Runnable.h<br>
│   │   ├── Thread.h<br>
│   │   └── Timer.h<br>
│   ├── panda<br>
│   │   ├── Chunk.h<br>
│   │   ├── Combiner.h<br>
│   │   ├── DataChunk.h<br>
│   │   ├── EmitConfiguration.h<br>
│   │   ├── FileChunk.h<br>
│   │   ├── FileOutput.h<br>
│   │   ├── KeyValueChunk.h<br>
│   │   ├── Mapper.h<br>
│   │   ├── MapReduceJob.h<br>
│   │   ├── Message.h<br>
│   │   ├── OutputAsInput.h<br>
│   │   ├── Output.h<br>
│   │   ├── PandaCPUConfig.h<br>
│   │   ├── PandaGPUConfig.h<br>
│   │   ├── PandaMapReduceJob.h<br>
│   │   ├── PandaMessage.h<br>
│   │   ├── PandaMessageIORequest.h<br>
│   │   ├── PandaMessagePackage.h<br>
│   │   ├── PandaMPIMessage.h<br>
│   │   ├── Partitioner.h<br>
│   │   └── Reducer.h<br>
│   ├── PandaAPI.h<br>
│   └── Panda.h<br>
├── LICENSE<br>
├── Makefile<br>
├── makefile.in<br>
├── README.md<br>
├── splitmpisendasanotherthread<br>
└── src<br>
    ├── cudacpp<br>
    │   └── Stream.cpp<br>
    ├── inputformat<br>
    │   ├── Chunk.cpp<br>
    │   ├── DataChunk.cpp<br>
    │   ├── FileChunk.cpp<br>
    │   └── KeyValueChunk.cpp<br>
    ├── message<br>
    │   ├── Messsage.cpp<br>
    │   ├── PandaMessage.cpp<br>
    │   ├── PandaMessageIORequest.cpp<br>
    │   └── PandaMPIMessage.cpp<br>
    ├── oscpp<br>
    │   ├── AsyncFileReader.cpp<br>
    │   ├── AsyncIORequest.cpp<br>
    │   ├── Condition.cpp<br>
    │   ├── Mutex.cpp<br>
    │   ├── Runnable.cpp<br>
    │   ├── Thread.cpp<br>
    │   └── Timer.cpp<br>
    ├── outputformat<br>
    │   └── Output.cpp<br>
    ├── pandajob<br>
    │   ├── Compare.cpp<br>
    │   ├── MapReduceJob.cpp<br>
    │   ├── PandaMapReduceJob.cpp<br>
    │   └── Partitioner.cpp<br>
    └── runtime<br>
        ├── PandaLib.cu<br>
        ├── PandaSched.cu<br>
        ├── PandaSort.cu<br>
        └── PandaUtils.cu<br>
