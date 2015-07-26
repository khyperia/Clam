#include "network.h"
#include "option.h"
#include "util.h"

Connection::Connection()
{
    std::string bindTo = HostBindPort();
    std::string connectTo = MasterIp();
    if (!bindTo.empty() && !connectTo.empty())
    {
        throw std::runtime_error("Specified both host bind port and master IP");
    }
    if (bindTo.empty() && connectTo.empty())
    {
        socket = NULL;
        socketSet = NULL;
        return;
    }
    if (bindTo.empty())
    {
        socketIsHost = false;
        size_t colonIndex = connectTo.find_last_of(':');
        if (colonIndex == std::string::npos)
        {
            throw std::runtime_error("Connect to IP missing port");
        }
        std::string host = connectTo.substr(0, colonIndex);
        std::string portStr = connectTo.substr(colonIndex + 1);
        Uint16 port = fromstring<Uint16>(portStr);
        IPaddress ip;
        if (SDLNet_ResolveHost(&ip, host.c_str(), port))
        {
            throw std::runtime_error(SDLNet_GetError());
        }
        socket = SDLNet_TCP_Open(&ip);
        if (!socket)
        {
            throw std::runtime_error("Failed to connect to " + connectTo);
        }
        socketSet = SDLNet_AllocSocketSet(1);
        SDLNet_AddSocket(socketSet, (SDLNet_GenericSocket) socket);
    }
    else
    {
        socketIsHost = true;
        Uint16 port = fromstring<Uint16>(bindTo);
        IPaddress ip;
        if (SDLNet_ResolveHost(&ip, NULL, port))
        {
            throw std::runtime_error(SDLNet_GetError());
        }
        socket = SDLNet_TCP_Open(&ip);
        if (!socket)
        {
            throw std::runtime_error("Failed to host at port " + bindTo);
        }
        socketSet = NULL;
    }
}

Connection::~Connection()
{
    if (socketSet)
    {
        SDLNet_FreeSocketSet(socketSet);
    }
    SDLNet_TCP_Close(socket);
    for (size_t i = 0; i < clients.size(); i++)
    {
        delete clients[i];
    }
}

void Connection::Send(const void *data, size_t size)
{
    if (SDLNet_TCP_Send(socket, data, (int) size) != (int) size)
    {
        throw std::runtime_error("Send on socket failed");
    }
}

void Connection::Recv(void *data, size_t size)
{
    if (SDLNet_TCP_Recv(socket, data, (int) size) != (int) size)
    {
        throw std::runtime_error("Recv on socket failed");
    }
}

void Connection::Sync(Kernel *kernel)
{
    if (socket == NULL)
    {
        return;
    }
    if (socketIsHost)
    {
        while (true)
        {
            TCPsocket newSocket = SDLNet_TCP_Accept(socket);
            if (!newSocket)
            {
                break;
            }
            clients.push_back(new Connection(newSocket));
        }
        for (size_t i = 0; i < clients.size(); i++)
        {
            kernel->SendState(clients[i]);
        }
    }
    else
    {
        while (SDLNet_CheckSockets(socketSet, 0) == 1)
        {
            kernel->RecvState(this);
        }
    }
}

class FileStateSync : public StateSync
{
    FILE *file;

    void Send(const void *data, size_t size)
    {
        fwrite(data, size, 1, file);
    }

    void Recv(void *data, size_t size)
    {
        fread(data, size, 1, file);
    }

    ~FileStateSync()
    {
        fclose(file);
    };

public:
    FileStateSync(const char *filename, const char *mode)
            : file(fopen(filename, mode))
    {
        if (!file)
        {
            throw std::runtime_error("Could not open file " + std::string(filename));
        }
    }
};

StateSync *NewFileStateSync(const char *filename, const char *mode)
{
    return new FileStateSync(filename, mode);
}
