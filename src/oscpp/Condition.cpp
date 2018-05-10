#include <oscpp/Condition.h>
#include <pthread.h>

namespace oscpp
{

  Condition::Condition()
  {
	  pthread_mutex_init((pthread_mutex_t*)mutex.getHandle(),NULL);
      	  handle= new pthread_cond_t;
	  pthread_cond_init((pthread_cond_t*)handle,NULL);
	  //pthread_cond_wait((pthread_cond_t*)handle, (pthread_mutex_t *)(mutex.getHandle()));
  }

  Condition::~Condition()
  {
	  delete[] (char *)handle;
  }

  void	Condition::signal()
  {
	  pthread_cond_signal((pthread_cond_t*)handle);
  }//void

  void	Condition::broadcast()
  {
	  pthread_cond_broadcast((pthread_cond_t*)handle);
  }//void

  void	Condition::wait()
  {

	  //pthread_cond_wait((pthread_cond_t*)handle, (pthread_mutex_t *)(mutex.getHandle()));
	  //pthread_cond_t test;
	  //pthread_cond_init((pthread_cond_t*)&test,NULL);
	  //pthread_mutex_t mut;
	  //pthread_mutex_init((pthread_mutex_t *)(mutex.getHandle()),NULL);
	  //(pthread_mutex_t *)(mutex.getHandle()) This is null todo
	  pthread_cond_wait((pthread_cond_t*)handle, (pthread_mutex_t *)(mutex.getHandle()));

  }//void
	
  void	Condition::lockMutex()
  {
	pthread_mutex_lock((pthread_mutex_t*)mutex.getHandle());
  }

  void	Condition::unlockMutex()
  {
	pthread_mutex_unlock((pthread_mutex_t*)mutex.getHandle());
  }
    
}
