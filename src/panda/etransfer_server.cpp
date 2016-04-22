#include <iostream>
#include <signal.h>

//#include "utransfer.h"
//#include "rusocket.h"
#include "datatype.h"
#include "PPTimers.h"
#include "eTransfer.h"

//tcp server
using namespace std;

int main(int argc, char** argv)
{

	CUtility::Sleep(100);
	eTransfer *transfer = new eTransfer();
	transfer->init_tcp(4323);
	UTcpSocket *tcp_listener = transfer->get_tcp_listener();
	list<upoll_t> src;
	list<upoll_t> dst;
	list<upoll_t>::iterator it;
	upoll_t ls_poll;
	ls_poll.events = UPOLL_READ_T;
	ls_poll.pointer = NULL;
	ls_poll.usock = tcp_listener;
	src.push_back(ls_poll);
	map<USocket*,upoll_t>m_tcp_map;
	struct timeval base_tm = {0,100};
	struct timeval wait_tm;
	//signal(SIGPIPE, SIG_IGN);
	while(true)
	{
		dst.clear();
		src.clear();
		map<USocket*,upoll_t>::iterator mit;
		for(mit = m_tcp_map.begin();mit!=m_tcp_map.end();mit++)
			src.push_back(mit->second);
		src.push_back(ls_poll);
		wait_tm = base_tm;
		//fprintf(stderr,"(debug invoke walk_through)\n");
		transfer->walk_through();
		int res = transfer->select(dst,src,&wait_tm);
		if (res>0)
		{
			for(it=dst.begin();it!=dst.end();it++)
			{
				upoll_t up = *it;
				if(up.usock == tcp_listener)
				{
					if(up.events&UPOLL_READ_T)
					{
						USocket *sc = transfer->accept(tcp_listener);
						if(sc)
						{
							cerr<<"up.events 1)UPOLL_READ_T 2)accept\n";
							upoll_t new_up;
							new_up.pointer = NULL;
							new_up.usock = sc;
							new_up.events = UPOLL_READ_T;
							m_tcp_map[sc] = new_up;
						}//if
					}
					//cerr<<"(up.sock == tcp_listener)\n";
					//it->usock->recv(data,1024,NULL);
				}else if(up.events & UPOLL_READ_T)
				{
					cerr<<"(up.events & UPOLL_READ_T)\n";
					char data[1024];
					int recv_size = it->usock->recv(data,1024,NULL);
					fprintf(stderr,"recv_size:%d\n",recv_size);
					return 0;
				}//else
				else if(up.events & UPOLL_WRITE_T)
				{
					cerr<<"(up.events & UPOLL_WRITE_T)\n";
					char data[1024];
					it->usock->send(data,1024,NULL);
					return 0;
				}else if(up.events & UPOLL_ERROR_T)
				{
					cerr<<"SYSTEM ERROR."<<endl;
					return 0;	
				}//else
			}
		}//res
		
	}

	return 0;
	//UTcpSocket *m_tcp_listener = new UTcpSocket();
	int m_tcp_port = 4323;	
	int m_tcp_sock = socket(PF_INET,SOCK_STREAM,0);
	struct sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = INADDR_ANY;
	addr.sin_port = htons(4323);
	if(bind(m_tcp_sock,(sockaddr *)&addr,sizeof(addr))<0)
	{
		cerr<<"bind error."<<endl;
	}else 
	{
		if(listen(m_tcp_sock,5)<0)
		cerr<<"listen error."<<endl;
	}//else

	//struct sockaddr_in addr;
        socklen_t addrlen = sizeof(addr);
	int new_sock = ::accept(m_tcp_sock, (struct sockaddr*)&addr,
		&addrlen);

        if (new_sock >= 0)
        {
                UTcpSocket *usocket = new UTcpSocket();
                CSockAddr caddr(addr);
                usocket->bind(new_sock, caddr);
		fprintf(stderr,"accept success\n");
		char rb[1024];
		int nbytes = usocket->recv(rb,1024,NULL);
		fprintf(stderr,"recv:%d\n",nbytes);	
                //m_main_mutex.on();
                //m_tcp_map[new_sock] = usocket;
                //m_tcp_accept_list.push_back(usocket);
                //m_main_mutex.off();
        }//if

	//int
	CPPTimers timers(4);
	timers.setTimer(0, 3000);
        timers.trigger(0);
	
	if (argc < 3)
	{
		printf("(%s listening port localfile)\n", argv[0]);
		return 1;
	}//if
	return 0;
}
