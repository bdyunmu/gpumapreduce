#include <oscpp/Thread.h>
#include <oscpp/Runnable.h>
#include <signal.h>
#include <errno.h>

namespace oscpp
{
	void * Thread::startThread(void * vself)
	{
		Runnable* pRunner = (Runnable*)(vself);	
		pRunner->run();
		printf("lihui +++++++++++++ Big Test startThread pRunner->run is done\n");
		return 0;
	};

	Thread::Thread(Runnable * const pRunner)
	{
		running = false;
		handle = (pthread_t *)malloc(sizeof(pthread_t));
		runner = pRunner;
	};

	Thread::~Thread(){};

	void Thread::yield()
	{
		
	};

	void Thread::start()
	{
	//	if (pthread_create((pthread_t*)handle,NULL,startThread, runner)!=0)
		if(pthread_create(&handle0,NULL,startThread, runner)!=0)
		perror("Thread creation failed!\n");
		running = true;
	};

	/*void Thread::run()
	{

	};*/

	void Thread::join()
	{
	#if 0
	printf("Big Test kill __________________________________________________________\n");
	int ret = pthread_kill(*(pthread_t*)handle,0);
   	if(ret == ESRCH)
        	printf("thread kill this thread does not exist or already exit.\n");
   	else if(ret == EINVAL)
        	printf("thread kill invoke useless msg\n");
   	else
        	printf("thread kill exist\n");
	#endif	
#if 1
	printf("Big Test join start 	___________________________________________________________\n");
	void *exitstat = NULL;
	if (pthread_join(handle0,NULL)!=0){
		printf("joining failed\n");
	}
	printf("Big Test join end	===========================================================\n");
	running = false;
#endif
	};

	bool Thread::isRunning()
	{
		return running;
	};

}//namespace


