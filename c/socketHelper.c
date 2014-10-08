#include "socketHelper.h"
#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#undef _POSIX_C_SOURCE
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

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
        if ((sock = socket(addr->ai_family, addr->ai_socktype | SOCK_CLOEXEC, addr->ai_protocol))
                == -1)
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
