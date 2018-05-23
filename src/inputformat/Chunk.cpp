#include <panda/Chunk.h>
#include <stdio.h>

namespace panda
{
  Chunk::Chunk()
  {
  }
  Chunk::~Chunk()
  {
  }

  MapTask::MapTask()
  {
	  keySize = 0;
	  valSize = 0;
	  key = NULL;
	  val = NULL;
  }

  MapTask::MapTask(int keySize, void *key, int valSize, void *val)
  {
	  this->key = key;
	  this->keySize = keySize;
	  this->val = val;
	  this->valSize = valSize;
  }

  MapTask::~MapTask(){}

}
