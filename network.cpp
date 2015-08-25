#include "network.h"
#include "option.h"

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
        SDLNet_AddSocket(socketSet, (SDLNet_GenericSocket)socket);
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
    int sentSize = SDLNet_TCP_Send(socket, data, (int)size);
    if (sentSize != (int)size)
    {
        throw std::runtime_error("Send on socket failed (attempted "
                                 + tostring(size) + ", actual " + tostring(sentSize) + ")");
    }
}

void Connection::Recv(void *data, size_t size)
{
    int recvSize = SDLNet_TCP_Recv(socket, data, (int)size);
    if (recvSize != (int)size)
    {
        throw std::runtime_error("Recv on socket failed (attempted "
                                 + tostring(size) + ", actual " + tostring(recvSize) + ")");
    }
}

// Returns true on failure
bool Connection::Sync(Kernel *kernel)
{
    if (socket == NULL)
    {
        return false;
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
            try
            {
                kernel->SendState(clients[i]);
            }
            catch (const std::runtime_error &e)
            {
                std::cout << "Sending state failed:" << std::endl;
                std::cout << e.what() << std::endl;
                return true;
            }
        }
    }
    else
    {
        while (SDLNet_CheckSockets(socketSet, 0) == 1)
        {
            try
            {
                kernel->RecvState(this);
            }
            catch (const std::runtime_error &e)
            {
                std::cout << "Receiving state failed:" << std::endl;
                std::cout << e.what() << std::endl;
                return true;
            }
        }
    }
    return false;
}

class FileStateSync : public StateSync
{
    FILE *file;
    bool reading;

    void Send(const void *data, size_t size)
    {
        if (reading)
        {
            throw std::runtime_error("Cannot Send a reading StateSync stream");
        }
        fwrite(data, size, 1, file);
    }

    void Recv(void *data, size_t size)
    {
        if (!reading)
        {
            throw std::runtime_error("Cannot Recv a writing StateSync stream");
        }
        fread(data, size, 1, file);
    }

    ~FileStateSync()
    {
        if (reading)
        {
            long current = ftell(file);
            fseek(file, 0, SEEK_END);
            long end = ftell(file);
            if (current != end)
            {
                throw std::runtime_error("");
            }
        }
        fclose(file);
    };

public:
    FileStateSync(const char *filename, bool reading)
            : file(fopen(filename, reading ? "rb" : "wb")), reading(reading)
    {
        if (!file)
        {
            throw std::runtime_error("Could not open file " + std::string(filename));
        }
    }
};

StateSync *NewFileStateSync(const char *filename, bool reading)
{
    return new FileStateSync(filename, reading);
}
