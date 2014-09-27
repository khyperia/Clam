#pragma once
#include <sys/socket.h>
#include <fcntl.h>
#include <netdb.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <algorithm>
#include <thread>
#include <limits>

enum MessageType : unsigned int
{
    MessageNull = 0,
    MessageOkay = 0,
    MessageKernelInvoke = 1,
    MessageKernelSource = 2,
    MessageMkBuffer = 3,
    MessageDlBuffer = 4,
    MessageRmBuffer = 5,
    MessageKill = std::numeric_limits<unsigned int>::max()
};

struct CSocket
{
    int sockfd;

    ~CSocket()
    {
        close(sockfd);
    }

    CSocket(CSocket const&) = delete;

    // connect constructor
    CSocket(const char* host, const char* port)
    {
        sockfd = mksock(host, port);
    }

    // host constructor (do not use Send/Recv! Use Accept())
    CSocket(const char* port)
    {
        sockfd = mksock(nullptr, port);
    }

    CSocket(int socketfd)
    {
        sockfd = socketfd;
    }

    private:
    static int mksock(const char* host, const char* port)
    {
        struct addrinfo hints, *servinfo = nullptr;
        memset(&hints, 0, sizeof hints);
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = host == nullptr ? AI_PASSIVE : 0;
        hints.ai_protocol = 0;
        hints.ai_canonname = nullptr;
        hints.ai_addr = nullptr;
        hints.ai_next = nullptr;

        int rv;
        if ((rv = getaddrinfo(host, port, &hints, &servinfo)) != 0) {
            throw std::runtime_error("getaddrinfo() failed: " + std::string(gai_strerror(rv)));
        }

        int tmpsockfd = -1;
        addrinfo* addressInformation = nullptr;
        for (auto p = servinfo; p != nullptr; p = p->ai_next)
        {
            if ((tmpsockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1)
            {
                puts(("socket() failed: " + std::string(gai_strerror(errno))).c_str());
            }
            else if (host == nullptr)
            {
                if (bind(tmpsockfd, p->ai_addr, p->ai_addrlen) == -1)
                {
                    puts(("bind() failed: " + std::string(gai_strerror(errno))).c_str());
                }
                else if (listen(tmpsockfd, 16) == -1)
                {
                    puts(("listen() failed: " + std::string(gai_strerror(errno))).c_str());
                }
                else
                {
                    addressInformation = p;
                    break;
                }
            }
            else 
            {
                if (connect(tmpsockfd, p->ai_addr, p->ai_addrlen) == -1)
                {
                    puts(("connect() failed: " + std::string(gai_strerror(errno))).c_str());
                    close(tmpsockfd);
                }
                else
                {
                    addressInformation = p;
                    break;
                }
            }
        }
        freeaddrinfo(servinfo);
        if (tmpsockfd == -1 || addressInformation == nullptr)
            throw std::runtime_error("No sockets succeeded in binding");
        return tmpsockfd;
    }

    static void runServer(std::shared_ptr<std::vector<int>> sockfds, int serverfd)
    {
        while (true)
        {
            int clientfd = accept(serverfd, nullptr, nullptr);
            if (clientfd == -1)
            {
                puts(("Server thread crashed on accept(): "
                            + std::string(gai_strerror(errno))).c_str());
                break;
            }
            sockfds->push_back(clientfd);
        }
    }

    static void runEcho(int myfd, std::vector<int> echoTo)
    {
        std::vector<uint8_t> buffer(64);
        std::vector<int> dead;
        while (true)
        {
            auto size = recv(myfd, buffer.data(), buffer.size(), 0);
            if (size == -1)
            {
                puts(("Echo thread crashed on recv(): " +
                            std::string(gai_strerror(errno))).c_str());
                close(myfd);
                break;
            }

            for (auto echo : echoTo)
            {
                auto result = send(echo, buffer.data(), size, 0);
                if (result != size)
                {
                    puts(("Echo thread failed to send(): " +
                                std::string(gai_strerror(errno))).c_str());
                    close(echo);
                    dead.push_back(echo);
                }
            }
            for (auto deadSock : dead)
                echoTo.erase(std::find(echoTo.begin(), echoTo.end(), deadSock));
            dead.clear();
        }
    }

    public:
    void SetNonblock()
    {
        if (fcntl(sockfd, F_SETFL, fcntl(sockfd, F_GETFL) | O_NONBLOCK) < 0)
        {
            throw std::runtime_error("SetNonblock() failed");
        }
    }

    std::shared_ptr<CSocket> Accept()
    {
        int clientfd = accept(sockfd, nullptr, nullptr);
        if (clientfd == -1)
            throw std::runtime_error("accept() failed: " + std::string(gai_strerror(errno)));
        return std::make_shared<CSocket>(clientfd);
    }

    template<typename T>
        void Send(std::vector<T> vector)
        {
            auto result = send(sockfd, vector.data(), vector.size() * sizeof(T), 0);
            if (result == -1 || (size_t)result != vector.size() * sizeof(T))
                throw std::runtime_error("send() failed: " +
                        std::string(gai_strerror(errno)));
        }

    template<typename T>
        std::vector<T> Recv(int length)
        {
            if (length == 0)
                return std::vector<T>();
            while (true)
            {
                auto result = RecvMaybe<T>(length);
                if (result.size() != 0)
                    return result;
            }
        }

    template<typename T>
        std::vector<T> RecvMaybe(int length)
        {
            std::vector<T> buf(length);
            int totalRead = 0;
            while (totalRead < length * (int)sizeof(T))
            {
                auto size = recv(sockfd, (uint8_t*)buf.data() + totalRead,
                        buf.size() * sizeof(T) - totalRead, 0);
                if ((size == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) || size == 0)
                {
                    if (totalRead == 0)
                        return std::vector<T>();
                }
                else if (size == -1)
                {
                    throw std::runtime_error("recvMaybe() failed: " +
                            std::string(gai_strerror(errno)));
                }
                if (size > 0)
                    totalRead += size;
            }
            return buf;
        }

    void SendStr(std::string str)
    {
        Send<unsigned int>({(unsigned int)str.length()});
        Send<char>(std::vector<char>(str.begin(), str.end()));
    }

    std::string RecvStr()
    {
        auto size = Recv<unsigned int>(1)[0];
        auto vec = Recv<char>(size);
        return std::string(vec.begin(), vec.end());
    }
};
