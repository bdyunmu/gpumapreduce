#ifndef _ETRANSFER_H
#define _ETRANSFER_H

#include <map>
#include <list>
#include <time.h>

#include "usocket.h"
#include "utcp_socket.h"

//#include "uudp_socket.h"
//#include "rusocket.h"
//#include "rupacket.h"
//#include "thread.h"

using std::map;
using std::list;

enum
{
	UPOLL_READ_T=1,
	UPOLL_WRITE_T=2,
	UPOLL_ERROR_T=4,
};

struct upoll_t
{
	USocket *usock;
	short events;
	void *pointer;
};

class eMutex{
public:
        pthread_mutex_t _mutex;
        pthread_mutex_t &get_mutex() {return _mutex;}
        eMutex();
        ~eMutex();
        void on(){
                pthread_mutex_lock(&_mutex);
        };
        void off(){
                pthread_mutex_unlock(&_mutex);
        };
};

class WEvent
{
public:
	WEvent()
	{
		pthread_cond_init(&cond,NULL);
	}
	WEvent(bool bManualReset)
	{
		pthread_cond_init(&cond,NULL);
	}
	~WEvent()
	{
		pthread_cond_destroy(&cond);
	}//WEvent
private:
	eMutex cs;
	pthread_cond_t cond;
};

class eTransfer
{

public:
	static eTransfer* get_instance();
	int init_tcp(int port);
	//int init_udp(int port);
	UTcpSocket *get_tcp_listener()
	{
		return m_tcp_listener;
	}//UTcpSocket
	int select(list<upoll_t>&dst,const list<upoll_t>&src,struct timeval*tm,WEvent *e=NULL);
	inline int has_available_sockets(list<upoll_t>&dst,const list<upoll_t>&src);
	inline int set_conn_set(fd_set& rset,fd_set&wset,fd_set& eset);
	inline int MY_TIMEVAL_TO_TIMESPEC(struct timeval *, struct timespec *);
	USocket * accept(USocket * usocket);
	eTransfer();
	~eTransfer();
	void walk_through();	
	int proc_tcp_listen_sock();
	int proc_tcp_socket(fd_set&rset,fd_set&wset,fd_set&eset,list<USocket*>&rlist,list<USocket*>&wlist);
	int proc_rw_set(fd_set&rset,fd_set&wset,fd_set&eset,list<USocket*>&rlist,list<USocket*>&wlist);
	void add_event(WEvent*e);	
	int do_destroy_sockets();
	int add_destroy_socket(USocket *usocket);
	int destroy_socket(USocket *usocket);
	int create_socket(USocket*&usocket,CSockAddr *addr, int type, void*param);
private:
	pthread_cond_t m_cond;
	int m_tcp_sock;
	int m_tcp_port;
	UTcpSocket *m_tcp_listener;
	map<int,UTcpSocket*>m_tcp_map;
	list<UTcpSocket*>m_tcp_accept_list;
	pthread_mutex_t m_wait_lock;
	eMutex m_main_mutex;
	eMutex m_send_mutex;
	struct timeval m_base_wait_time;	
	list<USocket*>m_destroy_list;
	bool m_terminated;
	list<WEvent *> m_event_list;
	eMutex m_event_list_mutex;
};

class AutoLock{
private:
	eMutex&mutex;
public:
	AutoLock(eMutex&_mutex):mutex(_mutex)
	{mutex.on();}
	~AutoLock()
	{mutex.off();}

};

#endif
