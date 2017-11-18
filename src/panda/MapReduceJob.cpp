#include <mpi.h>
#include <panda/Message.h>
#include <panda/Chunk.h>
#include <panda/MapReduceJob.h>
#include <cudacpp/Stream.h>

#include <oscpp/Condition.h>
#include <oscpp/Runnable.h>
#include <oscpp/Thread.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <list>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h> 
  #define popen _popen
  #define pclose _pclose
#endif

namespace panda
{

  void MapReduceJob::setDevice()
  {
    cudaSetDevice(0);
    MPI_Barrier(MPI_COMM_WORLD);
	return;

#ifdef _WIN32
    FILE * fp = popen("hostname.exe", "r");
#else
    FILE * fp = popen("/bin/hostname", "r");
#endif

    char buf[1024];
    if (fgets(buf, 1023, fp) == NULL)
		strcpy(buf, "localhost");

    pclose(fp);
    std::string host = buf;
    host = host.substr(0, host.size() - 1);
    strcpy(buf, host.c_str());
	
    int devCount;
	cudaGetDeviceCount(&devCount);

	cudaDeviceProp gpu_dev;
	cudaGetDeviceProperties(&gpu_dev, 0);
    ShowLog("There are %d  %s GPUs on host:%s",devCount, gpu_dev.name, buf);

	if (commRank == 0)
    {

      std::map<std::string, std::vector<int> > hosts;
      std::map<std::string, int> devCounts;
      MPI_Status stat;
      MPI_Request req;

      hosts[buf].push_back(0);
      devCounts[buf] = devCount;
      for (int i = 1; i < commSize; ++i)
      {
		MPI_Recv(&devCount, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);
        MPI_Recv(buf, 1024, MPI_CHAR, i, 1, MPI_COMM_WORLD, &stat);
        // check to make sure each process on each node reports the same number of devices.
        hosts[buf].push_back(i);
        if (devCounts.find(buf) != devCounts.end())
        {
          if (devCounts[buf] != devCount)
          {
            ShowError("Error, device count mismatch %d != %d on %s\n", devCounts[buf], devCount, buf); 
			//fflush(stdout);
          }//if
        }
        else devCounts[buf] = devCount;
      }
      // check to make sure that we don't have more jobs on a node than we have GPUs.
      for (std::map<std::string, std::vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it)
      {
        if (it->second.size() > static_cast<unsigned int>(devCounts[it->first]))
        {
          ShowError("Error, more jobs running on '%s' than devices - %d jobs > %d devices.\n",
                 it->first.c_str(), static_cast<int>(it->second.size()), devCounts[it->first]);
          //fflush(stdout);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }//if
      }//for
#if 0 // print out the configuration
      for (std::map<std::string, std::vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it)
      {
        printf("%s - %d\n", it->first.c_str(), devCounts[it->first]);
        for (unsigned int i = 0; i < it->second.size(); ++i) printf("  %d\n", it->second[i]);
      }
      fflush(stdout);
#endif

      // send out the device number for each process to use.
      MPI_Irecv(&deviceNum, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &req);
	  int devID = 0;
      for (std::map<std::string, std::vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it)
      {
        for (unsigned int i = 0; i < it->second.size(); ++i)
        {
          MPI_Send(&devID, 1, MPI_INT, it->second[i], 3, MPI_COMM_WORLD);
        }//for
      }//for
      MPI_Wait(&req, &stat);
    }
    else
    {
      //send out the hostname and device count for your local node, then get back the device number you should use.
      MPI_Status stat;
	  MPI_Send(&devCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      //MPI_Send(buf, strlen(buf) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	  MPI_Send(buf, 1024, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

      MPI_Recv(&deviceNum, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &stat);
    }//else

#if 0 // print out stuff
    MPI_Barrier(MPI_COMM_WORLD);
    printf("%d %s - using device %d (getDevice returns %d).\n", commRank, host.c_str(), deviceNum, cudacpp::Runtime::getDevice()); fflush(stdout);
#endif
    //cudacpp::Runtime::setDevice(deviceNum);
	ShowLog("cudaSetDevice [%d]",deviceNum);
	if (deviceNum!=0)
		ShowError("deviceNum!=0");
	cudaSetDevice(deviceNum);
    MPI_Barrier(MPI_COMM_WORLD);
  }//MPI_Barrier(MPI_COMM_WORLD);


  MapReduceJob::MapReduceJob(int & argc, char **& argv)
    : messager(NULL), commRank(-1), commSize(-1), deviceNum(-1)
  {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    gCommRank = commRank;
    ShowLog("CommSize:%d",commSize);
    setDevice();
  }//MapReduceJob

  MapReduceJob::~MapReduceJob()
  {

    if (messager        != NULL) delete messager;

	MPI_Barrier(MPI_COMM_WORLD);
	ShowLog("destroy MapReduce job and invoke MPI_Finalize");
    MPI_Finalize();

  }//MapReduceJob
}
