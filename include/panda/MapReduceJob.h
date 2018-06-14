#ifndef __PANDA_SUPMAPREDUCEJOB_H__
#define __PANDA_SUPMAPREDUCEJOB_H__

#include <panda/Output.h>
namespace panda
{
  class Message;
  class Chunk;
  class Partitioner;
  //class Sorter;

  class MapReduceJob
  {
    protected:
      Message *messager;
      Partitioner *partition;
      Output *output;
      //Sorter *sorter;

      int commRank, commSize, deviceNum;
      void setDevice();

    public:
      MapReduceJob(int & argc, char **&argv);
      MapReduceJob(int & argc, char **&argv, const bool bl);
      virtual ~MapReduceJob();

      Message    *getMessage()        {return messager;}
      int        getDeviceNumber()    {return deviceNum;}
      void setMessage    (Message  *const pMessage)  {messager =	pMessage;}
      void setPartition  (Partitioner *const pPartition) {partition = pPartition;}
      void setOutput (Output *pOutput){output=pOutput;}
      virtual void addInput(Chunk * chunk) = 0;
      //virtual void addCPUMapTasks(Chunk *chunk) = 0;
      //virtual void addGPUMapTasks(Chunk *chunk) = 0;
      virtual void execute() = 0;
      //virtual void setEnableGPU(bool b) = 0;
      //virtual void setEnableCPU(bool b) = 0;
  };
}

#endif
