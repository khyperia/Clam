#include "socketHelper.h"
#include "helper.h"
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void* echoThread(void* arg)
{
    int socketLeft = ((int*)arg)[0];
    int socketRight = ((int*)arg)[1];
    free(arg);
    struct pollfd polls[2];
    const int bufsize = 256;
    unsigned char buf[bufsize];
    while (1)
    {
        polls[0].events = POLLIN;
        polls[1].events = POLLIN;
        polls[0].fd = socketLeft;
        polls[1].fd = socketRight;
        int result = poll(polls, 2, 10000); // 10 seconds
        if (result == -1)
        {
            perror("poll()");
            break;
        }
        if (result == 0)
        {
            puts("poll(): timeout");
            break;
        }
        int gotdata = 0;
        if (polls[0].revents & POLLIN)
        {
            int size = recv(socketLeft, buf, bufsize, MSG_DONTWAIT);
            if (size == -1)
            {
                perror("recv()");
                break;
            }
            if (size > 0)
            {
                gotdata = 1;
                if (PrintErr(send_p(socketRight, buf, size)))
                    break;
            }
        }
        if (polls[1].revents & POLLIN)
        {
            int size = recv(socketRight, buf, bufsize, MSG_DONTWAIT);
            if (size == -1)
            {
                perror("recv()");
                break;
            }
            if (size > 0)
            {
                gotdata = 1;
                if (PrintErr(send_p(socketLeft, buf, size)))
                    break;
            }
        }
        if (!gotdata)
        {
            puts("No data after poll() return (socket closed?)");
            break;
        }
    }
    close(socketLeft);
    close(socketRight);
    puts("Echo thread exiting");
    return NULL;
}

int main(int argc, char** argv)
{
    if (argc < 4 || argc % 2 != 0)
    {
        printf("Usage: %s [host port] [dest ip1] [dest port1] [dest ip2] ...", argv[0]);
        return -1;
    }
    int hostsocket = hostSocket(argv[1]);
    if (hostsocket == -1)
    {
        puts("Unable to host socket. Exiting.");
        return -1;
    }
    int numOtherClients = argc / 2 - 1;
    pthread_t threads[numOtherClients];
    for (int slaveIndex = 0; slaveIndex < numOtherClients; slaveIndex++)
    {
        int master = accept(hostsocket, NULL, NULL);
        if (master == -1)
        {
            perror("accept()");
            exit(-1);
        }

        int slave = connectSocket(argv[slaveIndex * 2 + 2], argv[slaveIndex * 2 + 3]);
        if (slave == -1)
        {
            perror("connect()");
            exit(-1);
        }

        int* argument = (int*)malloc(2 * sizeof(int));
        if (!argument)
        {
            puts("malloc() failed");
            exit(-1);
        }
        argument[0] = master;
        argument[1] = slave;
        if (PrintErr(pthread_create(&threads[slaveIndex], NULL, echoThread, argument)))
        {
            puts("Exiting.");
            exit(-1);
        }
    }

    close(hostsocket);

    puts("All connections made, echo server no longer accepting");

    for (int slaveIndex = 0; slaveIndex < numOtherClients; slaveIndex++)
    {
        if (PrintErr(pthread_join(threads[slaveIndex], NULL)))
        {
            puts("Exiting.");
            exit(-1);
        }
    }

    puts("All threads joined, echo server closing");

    return 0;
}
