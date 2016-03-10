#pragma once

class StateSync;

#include "kernel.h"
#include <SDL_net.h>
#include <vector>

class StateSync
{
    virtual void Send(const void *data, size_t size) = 0;

    virtual void Recv(void *data, size_t size) = 0;

public:
    virtual ~StateSync()
    { };

    template<typename T>
    void Send(const T &data)
    {
        Send(&data, sizeof(T));
    }

    /*
    template<typename T>
    void SendArr(const std::vector<T> &data)
    {
        Send(data.data(), data.size() * sizeof(T));
    }
    */

    template<typename T>
    void SendFrom(T *data, size_t count)
    {
        Send(data, count * sizeof(T));
    }

    template<typename T>
    T Recv()
    {
        T *data = (T *)alloca(sizeof(T));
        Recv(data, sizeof(T));
        return *data;
    }

    /*
    template<typename T>
    std::vector<T> RecvArr(size_t count)
    {
        std::vector<T> data(count);
        Recv(data.data(), data.size() * sizeof(T));
        return data;
    }
    */

    template<typename T>
    void RecvInto(T *data, size_t count)
    {
        Recv(data, count * sizeof(T));
    }

    template<typename T>
    bool RecvChanged(T &original)
    {
        T newval = Recv<T>();
        if (original != newval)
        {
            original = newval;
            return true;
        }
        return false;
    }
};

class Connection : public StateSync
{
    TCPsocket socket;
    SDLNet_SocketSet socketSet;
    bool socketIsHost;
    std::vector<Connection *> clients;

    Connection(TCPsocket forward) : socket(forward), socketSet(NULL)
    { };
public:

    Connection();

    ~Connection();

    void Send(const void *data, size_t size);

    void Recv(void *data, size_t size);

    bool Sync(Kernel *kernel);

    bool IsSyncing();
};

StateSync *NewFileStateSync(const char *filename, bool reading);
