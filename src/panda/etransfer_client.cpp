#include <iostream>

//#include "utransfer.h"
//#include "rusocket.h"
//#include "datatype.h"
#include "eTransfer.h"

using namespace std;
int buffer_len;
//utcpsocket client
int main(int argc, char** argv)
{
		UTcpSocket *socket = new UTcpSocket();
		string host("127.0.0.1");
		unsigned short port = atoi("4323");
		CSockAddr raddr(host, port);
		int re1 = socket->init(raddr);
		int re2 = socket->connect();
		if(re1==0&&re2==0){
		fprintf(stderr,"creat socket (127.0.0.1) (4323) success.\n");
		}else{
			int err = socket->get_reason();
			delete socket;
			return err;
		}//else
		char sb[1024];
		int nbytes = 0;
		if(socket->can_write()){
		nbytes = socket->send(sb,1024,NULL);
		}//if
		fprintf(stderr,"socket->send %d bytes\n",nbytes);	
		return 0;
}//
