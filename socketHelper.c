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

// if host is null, makes a hosting socket
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
        if ((sock = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol)) == -1)
        {
            perror("socket()");
            continue;
        }
        if (host == NULL)
        {
            int theTrue = 1;
            if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &theTrue, sizeof(theTrue)))
            {
                perror("setsockopt()");
                continue;
            }
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
    char* result = malloc_s((size_t)stringSize * sizeof(char));
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

// Sends data to all sockets;
int send_all(int* socket, void* data, size_t numBytes)
{
    int errcode = 0;
    while (*socket)
    {
        if (*socket == -1)
            continue;
        int err = PrintErr(send_p(*socket, data, numBytes));
        if (err)
        {
            close(*socket);
            *socket = -1;
            errcode = err;
        }
        socket++;
    }
    return errcode;
}

// Sends an enum MessageType to all sockets
int send_all_msg(int* socket, enum MessageType messageType)
{
    return send_all(socket, &messageType, sizeof(messageType));
}

// Sends a string to all sockets
int send_all_str(int* socket, const char* string)
{
    int errcode = 0;
    while (*socket)
    {
        if (*socket == -1)
            continue;
        int err = PrintErr(send_str(*socket, string));
        if (err)
        {
            close(*socket);
            *socket = -1;
            errcode = err;
        }
        socket++;
    }
    return errcode;
}

// Receives an int from all sockets and checks if it is zero
int recv_all(int* socket)
{
    int errcode = 0;
    while (*socket)
    {
        if (*socket == -1)
            continue;
        int result = 0;
        int err = PrintErr(recv_p(*socket, &result, sizeof(int)));
        if (err)
        {
            close(*socket);
            *socket = -1;
            errcode = err;
        }
        else if (result)
        {
            PrintErr(result && "Slave didn't respond with zero");
            close(*socket);
            *socket = -1;
            errcode = result;
        }
        socket++;
    }
    return errcode;
}
