#include "masterSocket.h"
#include <limits.h>
#include <poll.h>
#include <stdlib.h>
#include "helper.h"

int numWaitingSoftSync = 0;

// Response to a MessageSync dispatch to the slave.
int messageSync(void)
{
    // if true, then the slave sent more sync responses than dispatches to it
    if (numWaitingSoftSync <= 0)
    {
        puts("numWaitingSoftSync <= 0, shouldn't have happened");
        return -1;
    }
    numWaitingSoftSync--;
    return 0;
}

// This function is the response handler for an earlier request to the slave for MessageDlBuffer
// The first argument is a long that represents a function pointer to a handler that was specified
// when it was originally dispatched
int messageDlBuffer(TCPsocket socketFd)
{
    long functionPointer;
    if (PrintErr(recv_p(socketFd, &functionPointer, sizeof(functionPointer))))
        return -1;
    long bufferSize = 0;
    if (PrintErr(recv_p(socketFd, &bufferSize, sizeof(long))))
        return -1;

    if (PrintErr(bufferSize < 0))
    {
        printf("bufferSize = %ld\n", bufferSize);
        return -1;
    }

    float* data = malloc_s((size_t)bufferSize);
    if (PrintErr(recv_p(socketFd, data, (size_t)bufferSize)))
    {
        free(data);
        return -1;
    }

    if (PrintErr(((int (*)(float*, long))functionPointer)(data, bufferSize)))
    {
        free(data);
        return -1;
    }
    free(data);

    return 0;
}

// Similar to slaveSocket's parseMessage, but the meanings of the enums are different
// due to messages going slave->master rather than master->slave
int parseMessage(enum MessageType messageType, TCPsocket socketFd)
{
    // We don't handle all the cases that the slave does, so ignore the warning
    switch (messageType)
    {
        case MessageSync:
            return PrintErr(messageSync());
        case MessageDlBuffer:
            return PrintErr(messageDlBuffer(socketFd));
        default:
            printf("Unknown message type %d\n", messageType);
            return -1;
    }
}

void removeSocketH(TCPsocket* toRemove)
{
    while (*toRemove)
    {
        *toRemove = toRemove[1];
        toRemove++;
    }
}

// Run the master socket receive loop.
// This and functions it calls should be the only ones reading from the sockets
// This function is very similar to slaveSocket's slaveSocket() function
int masterSocketRecv(TCPsocket* socketFds)
{
    int socketCount = 0;
    for (TCPsocket* sock = socketFds; *sock; sock++)
        socketCount++;
    SDLNet_SocketSet set = SDLNet_AllocSocketSet(socketCount);
    for (TCPsocket* sock = socketFds; *sock; sock++)
        SDLNet_TCP_AddSocket(set, *sock);

    while (1)
    {
        int numSocks = SDLNet_CheckSockets(set, 0);
        if (numSocks == -1)
        {
            puts("SDLNet_CheckSockets errored");
            SDLNet_FreeSocketSet(set);
            return -1;
        }
        if (numSocks == 0)
            break;

        for (TCPsocket* sock = socketFds; *sock; sock++)
        {
            TCPsocket socketFd = *socketFds;

            if (!SDLNet_SocketReady(socketFd))
                continue;

            enum MessageType messageType;
            if (PrintErr(recv_p(socketFd, &messageType, sizeof(messageType))))
            {
                SDLNet_TCP_DelSocket(set, *sock);
                SDLNet_TCP_Close(*sock);
                removeSocketH(sock);
                continue;
            }

            int result = PrintErr(parseMessage(messageType, socketFd));

            if (result)
            {
                puts("Closing socket");
                SDLNet_TCP_DelSocket(set, *sock);
                SDLNet_TCP_Close(*sock);
                removeSocketH(sock);
                continue;
            }
        }
    }
    SDLNet_FreeSocketSet(set);
    return 0;
}
