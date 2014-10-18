#include "socketHelper.h"
#include "helper.h"
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Runs a single "echo server", which takes two sockets and echos all data between them
// `arg` is int[2], representing two sockets. Will be free()'d
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
        // wait for communication
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
        // TODO: Extract to method?
        int gotdata = 0;
        // see `if (!gotdata)` comment why this isn't just `if (poll[0].revents)`
        if (polls[0].revents & POLLIN)
        {
            ssize_t size = recv(socketLeft, buf, bufsize, MSG_DONTWAIT);
            if (size == -1)
            {
                perror("recv()");
                break;
            }
            if (size > 0)
            {
                gotdata = 1;
                if (PrintErr((int)send_p(socketRight, buf, (size_t)size)))
                    break;
            }
        }
        if (polls[1].revents & POLLIN)
        {
            ssize_t size = recv(socketRight, buf, bufsize, MSG_DONTWAIT);
            if (size == -1)
            {
                perror("recv()");
                break;
            }
            if (size > 0)
            {
                gotdata = 1;
                if (PrintErr(send_p(socketLeft, buf, (size_t)size)))
                    break;
            }
        }
        // usually happens when a socket is closed
        // (revents probably contains a disconnected flag or something, and not POLLIN)
        if (!gotdata)
        {
            puts("No data after poll() return (socket closed?)");
            break;
        }
    }
    close(socketLeft);
    close(socketRight);
    puts("Echo thread exiting");
    // Return value is accessable from pthread_join, but it's not used there
    return NULL;
}

// Runs the echo server.
// Note: Doesn't use env vars, which is weird.
int main(int argc, char** argv)
{
    // Throughout: pthread says that returning from main() kills all threads
    if (argc < 4 || argc % 2 != 0)
    {
        printf("Usage: %s [host port] [dest ip1] [dest port1] [dest ip2] ...", argv[0]);
        return EXIT_FAILURE;
    }
    int hostsocket = hostSocket(argv[1]);
    if (hostsocket == -1)
    {
        puts("Unable to host socket. Exiting.");
        return EXIT_FAILURE;
    }
    int numOtherClients = argc / 2 - 1;
    pthread_t threads[numOtherClients];
    for (int slaveIndex = 0; slaveIndex < numOtherClients; slaveIndex++)
    {
        // accept the incoming connection
        int master = accept(hostsocket, NULL, NULL);
        if (master == -1)
        {
            perror("accept()");
            return EXIT_FAILURE;
        }

        // get the outgoing connection
        int slave = connectSocket(argv[slaveIndex * 2 + 2], argv[slaveIndex * 2 + 3]);
        if (slave == -1)
        {
            PrintErr("connect()" && -1);
            return EXIT_FAILURE;
        }

        // boot the client thread
        int* argument = malloc_s(2 * sizeof(int));
        argument[0] = master;
        argument[1] = slave;
        if (PrintErr(pthread_create(&threads[slaveIndex], NULL, echoThread, argument)))
        {
            puts("Exiting.");
            return EXIT_FAILURE;
        }
    }

    close(hostsocket);

    puts("All connections made, echo server no longer accepting");

    // wait for all the threads to exit
    for (int slaveIndex = 0; slaveIndex < numOtherClients; slaveIndex++)
    {
        if (PrintErr(pthread_join(threads[slaveIndex], NULL)))
        {
            puts("Exiting.");
            return EXIT_FAILURE;
        }
    }

    puts("All threads joined, echo server closing");

    return EXIT_SUCCESS;
}
