#ifndef DH_USOCKET
#define DH_USOCKET

#include "NetAux.h"
#include "SockBase.h"

enum
{
	SOCK_TYPE_TCP = 1,
	SOCK_TYPE_UDP = 2,
	SOCK_TYPE_RUDP = 3, 
};

enum SOCK_STATUS
{
	SOCK_CONNECTING,
	SOCK_ACCEPTED,
	SOCK_CONNECTED,
	SOCK_CLOSED,
	SOCK_ERROR,
};

/**
 * i set four variable for the derived class,
 * m_status: for the current socket status; the status defined in this class;
 * m_reason: the error number for the last error; the error number defined by system;
 * m_sock: the sock id; for tcp and udp connection, it's the socket created by calling socket;
 * for reliable udp socket, it's set by the reliable socket;
 * m_type: the type for current socket. it's set by the construct method of the derived class;
 */

class UTransfer;
class USocket
{
public:
	USocket(int type)
		:m_type(type), m_sock(-1), m_status(SOCK_CLOSED), transfer(0)
	{
	}

	virtual ~USocket()
	{
	}

	virtual int connect()
	{
		return -1;
	}

	virtual ssize_t recv(char* buf, size_t len, CSockAddr* addr)
	{
		return -1;
	}

	virtual ssize_t send(const char* buf, size_t len, const CSockAddr* addr)
	{
		return -1;
	}

	virtual bool can_read()
	{
		return false;
	}

	virtual bool can_write()
	{
		return false;
	}

	virtual int close()
	{
		return -1;
	}

	int get_status()
	{
		return m_status;
	}

	int get_reason()
	{
		return m_reason;
	}

	int get_sock()
	{
		return m_sock;
	}

	virtual void set_status(SOCK_STATUS s)
	{
		m_status = s;
	}

	virtual bool is_error() = 0;

	int getType(){
		return m_type;
	}
	void set_utransfer(UTransfer *p){ transfer = p;};
	UTransfer* get_utransfer(){return transfer;};
public:

//	friend class UTransfer;

	//the type and id make this socket to be the only one in our system
	int m_type; // the socket type, such as tcp/reliable udp/udp and so on
	int m_sock; //the socket id.
	int m_status; //current socket status
	int m_reason; // the last error reason, equals the errno
protected:
	UTransfer *transfer;
};

#endif
