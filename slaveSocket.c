#include "slaveSocket.h"
#include "helper.h"
#include "openclHelper.h"

// This is used to make sure the master doesn't get too far ahead of the client
// softSync is for
int messageSync(TCPsocket socketFd)
{
    enum MessageType message = MessageSync;
    if (PrintErr(send_p(socketFd, &message, sizeof(message))))
        return -1;
    return 0;
}

int messageKernelSource(struct Interop* interop, TCPsocket socketFd)
{
    int numStrings = 0;
    if (PrintErr(recv_p(socketFd, &numStrings, sizeof(int))))
        return -1;
    char** sources = malloc_s((size_t)numStrings * sizeof(char*));
    for (int source = 0; source < numStrings; source++)
        sources[source] = recv_str(socketFd);

    if (interop->clContext.kernels)
        deleteClContext(interop->clContext);

    int result = PrintErr(newClContext(interop, sources, (cl_uint)numStrings));

    for (int source = 0; source < numStrings; source++)
        free(sources[source]);
    free(sources);

    return result;
}

int recvArgument(struct Interop* interop, TCPsocket socketFd, struct ScreenPos screenPos,
        const char* kernelName, cl_uint* argIndex, int* involvedBuf0)
{
    int arglen = 0;
    if (PrintErr(recv_p(socketFd, &arglen, sizeof(int))))
        return -1;
    if (arglen == 0)
        return 1; // Special return code for "done"
    else if (arglen == -1) // Special code for memory argument
    {
        int bufferName = 0;
        if (PrintErr(recv_p(socketFd, &bufferName, sizeof(int))))
        {
            return -1;
        }
        if (bufferName == 0)
            *involvedBuf0 = 1;
        cl_mem memory = getMem(*interop, bufferName, NULL);
        if (memory == NULL)
        {
            return -1;
        }
        if (PrintErr(setKernelArg(&interop->clContext, kernelName, *argIndex,
                &memory, sizeof(cl_mem))))
        {
            return -1;
        }
    }
    else if (arglen == -2) // Special code for x/y/width/height
    {
        if (PrintErr(setKernelArg(&interop->clContext, kernelName, (*argIndex)++,
                &screenPos.x, sizeof(int))))
        {
            return -1;
        }
        if (PrintErr(setKernelArg(&interop->clContext, kernelName, (*argIndex)++,
                &screenPos.y, sizeof(int))))
        {
            return -1;
        }
        if (PrintErr(setKernelArg(&interop->clContext, kernelName, (*argIndex)++,
                &screenPos.width, sizeof(int))))
        {
            return -1;
        }
        if (PrintErr(setKernelArg(&interop->clContext, kernelName, *argIndex,
                &screenPos.height, sizeof(int))))
        {
            return -1;
        }
    }
    else
    {
        unsigned char arg[arglen];
        if (PrintErr(recv_p(socketFd, arg, (size_t)arglen)))
        {
            return -1;
        }
        if (PrintErr(setKernelArg(&interop->clContext,
                kernelName, *argIndex, arg, (size_t)arglen)))
        {
            return -1;
        }
    }
    return 0;
}

int messageKernelInvoke(struct Interop* interop, TCPsocket socketFd, struct ScreenPos screenPos,
        int* involvedBuf0)
{
    char* kernelName = recv_str(socketFd);
    if (kernelName == NULL)
        return -1;
    long launchSizeLong[2];
    if (PrintErr(recv_p(socketFd, launchSizeLong, sizeof(launchSizeLong))))
    {
        free(kernelName);
        return -1;
    }
    if (launchSizeLong[0] == -1)
        launchSizeLong[0] = screenPos.width;
    if (launchSizeLong[1] == -1)
        launchSizeLong[1] = screenPos.height;
    size_t launchSize[] = {(size_t)launchSizeLong[0], (size_t)launchSizeLong[1]};
    for (cl_uint argIndex = 0; ; argIndex++)
    {
        int argResult = recvArgument(interop, socketFd, screenPos, kernelName, &argIndex,
                involvedBuf0);
        if (argResult == 1)
            break;
        else if (PrintErr(argResult /* recvArgument() */))
        {
            free(kernelName);
            return -1;
        }
    }
    int result = PrintErr(invokeKernel(&interop->clContext,
            *interop, kernelName, launchSize));

    free(kernelName);

    return result;
}

int messageMkBuffer(struct Interop* interop, TCPsocket socketFd, struct ScreenPos screenPos)
{
    int bufferId = 0;
    long bufferSize = 0;
    if (PrintErr(recv_p(socketFd, &bufferId, sizeof(int))) ||
            PrintErr(recv_p(socketFd, &bufferSize, sizeof(long))))
        return -1;
    if (PrintErr(allocMem(interop, bufferId, (size_t)bufferSize,
            (size_t)screenPos.width * (size_t)screenPos.height * 4 * 4)))
        return -1;
    return 0;
}

int messageRmBuffer(struct Interop* interop, TCPsocket socketFd)
{
    int bufferId = 0;
    if (PrintErr(recv_p(socketFd, &bufferId, sizeof(int))))
        return -1;
    freeMem(interop, bufferId);
    return 0;
}

int messageDlBuffer(struct Interop* interop, TCPsocket socketFd, struct ScreenPos screenPos)
{
    long userdata = 0;
    if (PrintErr(recv_p(socketFd, &userdata, sizeof(userdata))))
        return -1;
    int bufferId = 0;
    if (PrintErr(recv_p(socketFd, &bufferId, sizeof(int))))
        return -1;
    size_t memSize = 0;
    float* data = dlMem(*interop, bufferId, &memSize,
            (size_t)screenPos.width * (size_t)screenPos.height * 4 * 4);
    if (!data)
        return -1;
    long memSizeL = (long)memSize;
    enum MessageType dlbuffer = MessageDlBuffer;
    if (PrintErr(send_p(socketFd, &dlbuffer, sizeof(dlbuffer))) ||
            PrintErr(send_p(socketFd, &userdata, sizeof(long))) ||
            PrintErr(send_p(socketFd, &memSizeL, sizeof(long))) ||
            PrintErr(send_p(socketFd, data, memSize)))
    {
        free(data);
        return -1;
    }
    free(data);
    return 0;
}

int messageUplBuffer(struct Interop* interop, TCPsocket socketFd)
{
    int bufferId = 0;
    if (PrintErr(recv_p(socketFd, &bufferId, sizeof(int))))
        return -1;
    long memSize = 0;
    if (PrintErr(recv_p(socketFd, &memSize, sizeof(long))))
        return -1;
    float* data = malloc_s((size_t)memSize);
    if (PrintErr(recv_p(socketFd, data, (size_t)memSize)))
    {
        free(data);
        return -1;
    }
    if (PrintErr(uplMem(interop, bufferId, (size_t)memSize, data)))
    {
        free(data);
        return -1;
    }
    free(data);
    return 0;
}

int parseMessage(enum MessageType messageType, struct Interop* interop,
        TCPsocket socketFd, struct ScreenPos screenPos, int* involvedBuf0)
{
    switch (messageType)
    {
        case MessageTerm:
            puts("MessageTerm caught in socket.");
            exit(EXIT_SUCCESS);
        case MessageSync:
            return PrintErr(messageSync(socketFd));
        case MessageKernelSource:
            return PrintErr(messageKernelSource(interop, socketFd));
        case MessageKernelInvoke:
            return PrintErr(messageKernelInvoke(interop, socketFd, screenPos, involvedBuf0));
        case MessageMkBuffer:
            return PrintErr(messageMkBuffer(interop, socketFd, screenPos));
        case MessageRmBuffer:
            return PrintErr(messageRmBuffer(interop, socketFd));
        case MessageDlBuffer:
            return PrintErr(messageDlBuffer(interop, socketFd, screenPos));
        case MessageUplBuffer:
            return PrintErr(messageUplBuffer(interop, socketFd));
        default:
            printf("Unknown message type %d\n", messageType);
            return -1;
    }
}

int slaveSocket(struct Interop* interop, TCPsocket socketFd, struct ScreenPos screenPos)
{
    SDLNet_SocketSet set = SDLNet_AllocSocketSet(1);
    SDLNet_TCP_AddSocket(set, socketFd);
    while (1)
    {
        int check = SDLNet_CheckSockets(set, 0);
        if (check == -1)
        {
            printf("SDLNet_CheckSockets: %s\n", SDLNet_GetError());
            SDLNet_FreeSocketSet(set);
            return -1;
        }
        if (check == 0)
        {
            SDLNet_FreeSocketSet(set);
            break;
        }

        enum MessageType messageType;
        if (PrintErr(recv_p(socketFd, &messageType, sizeof(messageType))))
            return -1;

        int involvedBuf0 = 0;
        int result = PrintErr(parseMessage(messageType, interop, socketFd, screenPos, &involvedBuf0));

        if (result)
            return -1;

        if (involvedBuf0)
        {
            // Give a chance for the slave to update the screen in a situation where commands are
            // coming in faster than execution
            // (Update screen if special buffer 0 was involved)
            break;
        }
    }
    return 0;
}
