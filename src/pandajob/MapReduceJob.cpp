#include <mpi.h>

#include <panda/Message.h>
#include <panda/Chunk.h>
#include <panda/MapReduceJob.h>
#include <panda/Partitioner.h>

#include <cudacpp/Stream.h>
#include <oscpp/Condition.h>
#include <oscpp/Runnable.h>
#include <oscpp/Thread.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace panda
{

  void MapReduceJob::setDevice()
  {
    cudaSetDevice(0);
    return;

  }//MPI_Barrier(MPI_COMM_WORLD);


  MapReduceJob::MapReduceJob(int & argc, char **& argv)
    : messager(NULL),partition(NULL),commRank(-1),commSize(-1),deviceNum(-1)
  {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    gCommRank = commRank;
    setDevice();
    partition = new Partitioner();
  }//MapReduceJob

  MapReduceJob::~MapReduceJob()
  {

    if (messager  != NULL) delete messager;
    if (partition != NULL) delete partition;
    ShowLog("destroy MapReduce job and invoke MPI_Finalize");
  }//MapReduceJob
}
