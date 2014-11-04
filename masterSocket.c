#include "masterSocket.h"
#include <poll.h>
#include <stdlib.h>
#include "helper.h"
#include "socketHelper.h"

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
int messageDlBuffer(int socketFd)
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
int parseMessage(enum MessageType messageType, int socketFd)
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

// Run the master socket receive loop.
// This and functions it calls should be the only ones reading from the sockets
// This function is very similar to slaveSocket's slaveSocket() function
int masterSocketRecv(int* socketFds)
{
    for (;*socketFds; socketFds++)
    {
        if (*socketFds == -1)
            continue;
        int socketFd = *socketFds;
        while (1)
        {
            struct pollfd poll_fd;
            poll_fd.events = POLLIN;
            poll_fd.fd = socketFd;
            int pollResult = poll(&poll_fd, 1, 0);
            if (PrintErr(pollResult == -1 && "poll()"))
                return -1;
            if (pollResult == 0 || !(poll_fd.revents & POLLIN))
                break;
            if (!(poll_fd.revents & POLLIN))
            {
                printf("Unknown pollfd.revents %d\n", poll_fd.revents);
                return -1;
            }

            enum MessageType messageType;
            if (PrintErr(recv_p(socketFd, &messageType, sizeof(messageType))))
                return -1;

            int result = PrintErr(parseMessage(messageType, socketFd));

            if (result)
                return -1;
        }
    }
    return 0;
}
