#pragma once

#include <stddef.h>

enum MessageType
{
    MessageTerm,
    MessageKernelInvoke,
    MessageKernelSource,
    MessageMkBuffer,
    MessageRmBuffer,
    MessageDlBuffer,
    MessageUplBuffer
};

int hostSocket(const char* port);
int connectSocket(const char* host, const char* port);
int recv_p(int socketFd, void* data, size_t numBytes);
int send_p(int socketFd, const void* data, size_t numBytes);
// must call returned char* with free()
char* recv_str(int socketFd);
int send_str(int socketFd, const char* string);
