#pragma once

enum MessageType
{
    MessageNull,
    MessageKernelInvoke,
    MessageKernelSource,
    MessageMkBuffer,
    MessageRmBuffer,
    MessageDlBuffer
};

int hostSocket(const char* port);
int connectSocket(const char* host, const char* port);
int recv_p(int socketFd, void* data, int numBytes);
int send_p(int socketFd, const void* data, int numBytes);
// must call returned char* with free()
char* recv_str(int socketFd);
int send_str(int socketFd, const char* string);
