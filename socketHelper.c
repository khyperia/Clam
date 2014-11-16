#include "socketHelper.h"
#include "helper.h"
#include <string.h>
#include <limits.h>

TCPsocket hostSocket(Uint16 port)
{
    IPaddress ip;
    if (PrintErr(SDLNet_ResolveHost(&ip, NULL, port)))
    {
        puts(SDLNet_GetError());
        return NULL;
    }
    TCPsocket result = SDLNet_TCP_Open(&ip);
    if (!result)
    {
        printf("SDLNet_TCP_Open: %s\n", SDLNet_GetError());
        return NULL;
    }
    return result;
}

TCPsocket connectSocket(const char* host, Uint16 port)
{
    IPaddress ip;
    if (PrintErr(SDLNet_ResolveHost(&ip, host, port)))
    {
        puts(SDLNet_GetError());
        return NULL;
    }
    TCPsocket result = SDLNet_TCP_Open(&ip);
    if (!result)
    {
        printf("SDLNet_TCP_Open: %s\n", SDLNet_GetError());
        return NULL;
    }
    return result;
}

int recv_p(TCPsocket socketFd, void* data, size_t numBytes)
{
    size_t read = 0;
    while (read < numBytes)
    {
        int numBytesThis = INT_MAX;
        if (numBytes - read < (size_t)numBytesThis)
            numBytesThis = (int)(numBytes - read);
        int result = SDLNet_TCP_Recv(socketFd, (char*)data + read, numBytesThis);
        if (result == -1)
        {
            printf("SDLNet_TCP_Recv(%d) == %d: %s\n", numBytesThis, result, SDLNet_GetError());
            return -1;
        }
        read += (size_t)result;
    }
    return 0;
}

int send_p(TCPsocket socketFd, const void* data, size_t numBytes)
{
    size_t read = 0;
    while (read < numBytes)
    {
        int numBytesThis = INT_MAX;
        if (numBytes - read < (size_t)numBytesThis)
            numBytesThis = (int)(numBytes - read);
        int result = SDLNet_TCP_Send(socketFd, (char*)data + read, numBytesThis);
        if (result == -1)
        {
            printf("SDLNet_TCP_Send(%d) == %d: %s\n", numBytesThis, result, SDLNet_GetError());
            return -1;
        }
        read += (size_t)result;
    }
    return 0;
}

char* recv_str(TCPsocket socketFd)
{
    unsigned long size;
    if (PrintErr(recv_p(socketFd, &size, sizeof(size))))
        return NULL;
    char* result = malloc_s((size_t)size + 1);
    if (PrintErr(recv_p(socketFd, result, size)))
    {
        free(result);
        return NULL;
    }
    result[size] = '\0';
    return result;
}

int send_str(TCPsocket socketFd, const char* string)
{
    unsigned long len = strlen(string);
    if (PrintErr(send_p(socketFd, &len, sizeof(len))))
        return -1;
    if (PrintErr(send_p(socketFd, string, len)))
        return -1;
    return 0;
}

void removeSocket(TCPsocket* toRemove)
{
    while (*toRemove)
    {
        *toRemove = toRemove[1];
        toRemove++;
    }
}

int send_all(TCPsocket* sockets, void* data, size_t numBytes)
{
    for (TCPsocket* sock = sockets; *sock; sock++)
    {
        if (PrintErr(send_p(*sock, data, numBytes)))
        {
            puts("Disconnecting client and ignoring error");
            SDLNet_TCP_Close(*sock);
            removeSocket(sock);
            sock--;
        }
    }
    return 0;
}

int send_all_msg(TCPsocket* sockets, enum MessageType messageType)
{
    return PrintErr(send_all(sockets, &messageType, sizeof(messageType)));
}

int send_all_str(TCPsocket* sockets, const char* string)
{
    for (TCPsocket* sock = sockets; *sock; sock++)
    {
        if (PrintErr(send_str(*sock, string)))
        {
            puts("Disconnecting client and ignoring error");
            SDLNet_TCP_Close(*sock);
            removeSocket(sock);
            sock--;
        }
    }
    return 0;
}
