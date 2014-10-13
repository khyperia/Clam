#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#undef _POSIX_C_SOURCE
#include "socketHelper.h"
#include "helper.h"
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>

int hostSocket(const char* port)
{
    return connectSocket(NULL, port);
}

int connectSocket(const char* host, const char* port)
{
    struct addrinfo hints, *servinfo = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = host == NULL ? AI_PASSIVE : 0;
    hints.ai_protocol = 0;
    hints.ai_canonname = NULL;
    hints.ai_addr = NULL;
    hints.ai_next = NULL;

    int rv;
    if ((rv = getaddrinfo(host, port, &hints, &servinfo)) != 0) {
        printf("getaddrinfo(): %s\n", gai_strerror(rv));
        return -1;
    }

    int sock = -1;
    struct addrinfo* addr = NULL;
    for (addr = servinfo; addr != NULL; addr = addr->ai_next)
    {
        if ((sock = socket(addr->ai_family,
                        addr->ai_socktype | SOCK_CLOEXEC, addr->ai_protocol)) == -1)
        {
            perror("socket()");
            continue;
        }
        if (host == NULL)
        {
            if (bind(sock, addr->ai_addr, addr->ai_addrlen) == -1)
            {
                perror("bind()");
                continue;
            }
            if (listen(sock, 16) == -1)
            {
                perror("listen()");
                continue;
            }
            break;
        }
        else
        {
            if (connect(sock, addr->ai_addr, addr->ai_addrlen) == -1)
            {
                perror("connect()");
                close(sock);
                continue;
            }
            break;
        }
    }
    if (addr == NULL)
        return -1;
    freeaddrinfo(servinfo);
    return sock;
}

int recv_p(int socketFd, void* data, size_t numBytes)
{
    if (socketFd == -1)
    {
        puts("recv(): Bad socket file descriptor");
        return -1;
    }
    ssize_t result = recv(socketFd, data, numBytes, MSG_WAITALL);
    if (result == -1)
    {
        perror("recv()");
        return -1;
    }
    if ((size_t)result != numBytes)
    {
        printf("recv(): not enough bytes read (%lu of %lu)\n", result, numBytes);
        return -1;
    }
    return 0;
}

int send_p(int socketFd, const void* data, size_t numBytes)
{
    size_t numBytesSent = 0;
    while (numBytesSent < numBytes)
    {
        ssize_t result = send(socketFd, data, numBytes, 0);
        if (result == -1)
        {
            perror("recv()");
            return -1;
        }
        if (result == 0)
        {
            printf("send(): not enough bytes sent (%lu of %lu)\n", numBytesSent, numBytes);
            return -1;
        }
        numBytesSent += (size_t)result;
    }
    return 0;
}

char* recv_str(int socketFd)
{
    int stringSize = 0;
    if (PrintErr(recv_p(socketFd, &stringSize, sizeof(int))))
        return NULL;
    char* result = malloc((size_t)stringSize * sizeof(char));
    if (!result)
    {
        puts("malloc() failed");
        exit(-1);
    }
    if (PrintErr(recv_p(socketFd, result, (size_t)stringSize)))
    {
        free(result);
        return NULL;
    }
    return result;
}

int send_str(int socketFd, const char* string)
{
    int len = (int)strlen(string) + 1;
    if (PrintErr(send_p(socketFd, &len, sizeof(int))))
        return -1;
    if (PrintErr(send_p(socketFd, string, (size_t)len)))
        return -1;
    return 0;
}
