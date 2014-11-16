#pragma once

#include <SDL2/SDL_net.h>

enum MessageType
{
    MessageTerm,
    MessageSync,
    MessageKernelInvoke,
    MessageKernelSource,
    MessageMkBuffer,
    MessageRmBuffer,
    MessageDlBuffer,
    MessageUplBuffer
};

TCPsocket hostSocket(Uint16 port);
TCPsocket connectSocket(const char* host, Uint16 port);

int recv_p(TCPsocket socketFd, void* data, size_t numBytes);
int send_p(TCPsocket socketFd, const void* data, size_t numBytes);
// must call returned char* with free()
char* recv_str(TCPsocket socketFd);
int send_str(TCPsocket socketFd, const char* string);

// these close and remove socket from list if they die
int send_all(TCPsocket* sockets, void* data, size_t numBytes);
int send_all_msg(TCPsocket* sockets, enum MessageType messageType);
int send_all_str(TCPsocket* sockets, const char* string);
