#ifndef UTCP_SOCKET
#define UTCP_SOCKET

#include "usocket.h"

class UTcpSocket:public USocket
{
public:
	UTcpSocket();
	virtual ~UTcpSocket();

	int init(const CSockAddr& remote_addr);
	int bind(int sock, const CSockAddr& addr);
	int connect();
	int register_sock(int sock);
	ssize_t recv(char* buf, size_t len, CSockAddr *addr);
	ssize_t send(const char *buf, size_t len, const CSockAddr *addr);
	bool can_read();
	bool can_write();
	int close();
	virtual bool is_error();
	bool m_bneedreadselect;	//only for tcp
	bool m_bneedwriteselect; //only for tcp

private:
	CSockAddr m_addr;

};

#endif
