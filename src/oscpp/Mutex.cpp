#include <oscpp/Mutex.h>
#include <pthread.h>

namespace oscpp
{
  Mutex::Mutex() 
  {
	 handle = (new pthread_mutex_t);
	 pthread_mutex_init((pthread_mutex_t*)handle,NULL);
  }//Mutex

  Mutex::~Mutex() 
  {
	 delete[]  (char *)handle;
  }//Mutex

  bool Mutex::tryLock()
  {
	int ret = pthread_mutex_trylock((pthread_mutex_t*)handle);
	if (ret == 0) return true;
	else    return false;
  }//bool

  void Mutex::lock()
  {
	pthread_mutex_lock((pthread_mutex_t*)handle);
  }//void

  void Mutex::unlock()
  {
	pthread_mutex_unlock((pthread_mutex_t*)handle);
  }//void

  void* Mutex::getHandle()
  {
	return handle;
  }//void

}
