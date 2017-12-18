#ifndef __PANDA_SUPMAPREDUCEJOB_H__
#define __PANDA_SUPMAPREDUCEJOB_H__

#include <vector>
namespace panda
{
  class Message;
  class Chunk;
  class MapReduceJob
  {
    protected:
      Message         * messager;
      int commRank, commSize, deviceNum;
      void setDevice();

    public:
      MapReduceJob(int & argc, char **&argv);
      MapReduceJob(int & argc, char **&argv, const bool bl);
      virtual ~MapReduceJob();
      inline Message          * getMessage()        { return messager;        }
      inline int               getDeviceNumber()    { return deviceNum;       }
      inline void setMessage       (Message       * const pMessage)         { messager       =	pMessage;        }
      virtual void addInput(Chunk * chunk) = 0;
      virtual void addCPUMapTasks(Chunk *chunk) = 0;
      virtual void addGPUMapTasks(Chunk *chunk) = 0;
      virtual void execute() = 0;
      virtual void setEnableGPU(bool b) = 0;
      virtual void setEnableCPU(bool b) = 0;
  };
}

#endif
