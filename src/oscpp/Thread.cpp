#include <oscpp/Thread.h>
#include <oscpp/Runnable.h>

namespace oscpp
{
	void * Thread::startThread(void * vself)
	{
		Runnable* pRun = (Runnable*)(vself);	
		pRun->run();
		return 0;
	};

	Thread::Thread(Runnable * const pRunner)
	{
		running = false;
		handle = (pthread_t *)malloc(sizeof(pthread_t));
		runner = pRunner;
		
		//runner->run = pRunner->run;
	};

	Thread::~Thread(){};

	void Thread::yield()
	{
		
	};

	void Thread::start()
	{
		if (pthread_create((pthread_t*)handle,NULL,startThread, runner)!=0)
			perror("Thread creation failed!\n");
		running = true;
	};

	void Thread::run()
	{

	};

	void Thread::join()
	{
		void *exitstat;
		if (pthread_join(*(pthread_t*)handle,&exitstat)!=0) 
			perror("joining failed\n");
	};

	bool Thread::isRunning()
	{
		return running;
	};

}//namespace


