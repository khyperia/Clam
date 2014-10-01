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
        if (close(sockfd))
            puts("WARNING: close(socket) returned nonzero, things may fail");
    }

    CSocket(CSocket const&) = delete;

    // connect constructor (parse host:port syntax)
    CSocket(std::string ip)
    {
        std::string port = "23456";
        auto colon = ip.find(":");
        if (colon != std::string::npos)
        {
            port = ip.substr(colon + 1, ip.length() - colon - 1);
            ip = ip.substr(0, colon);
        }
        sockfd = mksock(ip.c_str(), port.c_str());
    }

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

    int GetFd()
    {
        return sockfd;
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
            if ((tmpsockfd = socket(p->ai_family, p->ai_socktype | SOCK_CLOEXEC,
                            p->ai_protocol)) == -1)
            {
                puts(("socket() failed: " + std::string(gai_strerror(errno))).c_str());
            }
            else if (host == nullptr)
            {
                if (bind(tmpsockfd, p->ai_addr, p->ai_addrlen) == -1)
                {
                    puts(("bind() failed: " + std::string(strerror(errno))).c_str());
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

    public:
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
            auto result = send(sockfd, vector.data(), vector.size() * sizeof(T), MSG_NOSIGNAL);
            if (result == -1)
                throw std::runtime_error("send() failed: " +
                        std::string(gai_strerror(errno)));
            if (static_cast<size_t>(result) != vector.size() * sizeof(T))
                throw std::runtime_error("send() didn't send all bytes: " +
                        std::string(gai_strerror(errno)));
        }

    template<typename T>
        std::vector<T> Recv(int length)
        {
            if (length == 0)
                return std::vector<T>();
            std::vector<T> buf(static_cast<size_t>(length));
            int lengthbytes = static_cast<int>(buf.size() * sizeof(T));
            auto size = recv(sockfd, reinterpret_cast<uint8_t*>(buf.data()),
                    static_cast<size_t>(lengthbytes), MSG_WAITALL);
            if (size != lengthbytes)
            {
                if (size == -1)
                    throw std::runtime_error("recv() failed: " +
                            std::string(gai_strerror(errno)));
                throw std::runtime_error("recv() failed: Not enough bytes read (" +
                        std::to_string(size) + " of " +
                        std::to_string(lengthbytes) + ")");
            }
            return buf;
        }

    template<typename T>
        std::vector<T> RecvNoFull(int maxlength)
        {
            std::vector<T> buf(maxlength);
            auto size = recv(sockfd, reinterpret_cast<uint8_t*>(buf.data()),
                    buf.size() * sizeof(T), MSG_DONTWAIT);
            if ((size == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) || size == 0)
                return std::vector<T>();
            else if (size == -1)
                throw std::runtime_error("recvMaybe() failed: " +
                        std::string(gai_strerror(errno)));
            buf.resize(size);
            return buf;
        }

    template<typename T>
        std::vector<T> RecvMaybe(int length)
        {
            std::vector<T> buf(static_cast<size_t>(length));
            auto size = recv(sockfd, reinterpret_cast<uint8_t*>(buf.data()),
                    buf.size() * sizeof(T), MSG_DONTWAIT);
            if ((size == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) || size == 0)
                return std::vector<T>();
            if (size == -1)
                throw std::runtime_error("recvMaybe() failed: " +
                        std::string(gai_strerror(errno)));
            auto newsize = recv(sockfd, reinterpret_cast<uint8_t*>(buf.data()) + size,
                    buf.size() * sizeof(T) - static_cast<size_t>(size), MSG_WAITALL);
            if (newsize == -1)
                throw std::runtime_error("recvMaybe() failed: " +
                        std::string(gai_strerror(errno)));
            if (size + newsize != static_cast<int>(buf.size() * sizeof(T)))
                throw std::runtime_error("recvMaybe() failed: Not enough bytes read (" +
                        std::to_string(size + newsize) + " of " +
                        std::to_string(buf.size() * sizeof(T)) + ")");
            return buf;
        }

    bool RecvByte(uint8_t* byte)
    {
        auto result = recv(sockfd, byte, 1, MSG_DONTWAIT);
        if ((result == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) || result == 0)
            return false;
        if (result == -1)
            throw std::runtime_error("recvByte() failed: " +
                    std::string(gai_strerror(errno)));
        return true;
    }

    void SendByte(uint8_t byte)
    {
        if (send(sockfd, &byte, 1, MSG_NOSIGNAL) != 1)
            throw std::runtime_error("SendByte() failed: " +
                    std::string(gai_strerror(errno)));
    }

    void SendStr(std::string str)
    {
        Send<unsigned int>({static_cast<unsigned int>(str.length())});
        Send<char>(std::vector<char>(str.begin(), str.end()));
    }

    std::string RecvStr()
    {
        auto size = Recv<unsigned int>(1)[0];
        auto vec = Recv<char>(static_cast<int>(size));
        return std::string(vec.begin(), vec.end());
    }
};
