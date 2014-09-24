#include <boost/asio.hpp>

enum MessageType : unsigned int
{
    MessageNull = 0,
    MessageOkay = 0,
    MessageKernelInvoke = 1,
    MessageKernelSource = 2,
    MessageKill = UINT_MAX
};

    template<typename T>
void send(boost::asio::ip::tcp::socket& socket, std::vector<T> vector)
{
    socket.send(boost::asio::buffer(vector));
}

    template<typename T>
std::vector<T> read(boost::asio::ip::tcp::socket& socket, int numElements)
{
    std::vector<T> result(numElements);
    socket.receive(boost::asio::buffer(result));
    return result;
}

inline
std::string readStr(boost::asio::ip::tcp::socket& socket)
{
    unsigned int size = read<unsigned int>(socket, 1)[0];
    auto result = read<char>(socket, size);
    return std::string(result.data(), size);
}

inline
void writeStr(boost::asio::ip::tcp::socket& socket, std::string str)
{
    unsigned int length = str.size();
    socket.send(boost::asio::buffer(&length, sizeof(unsigned int)));
    socket.send(boost::asio::buffer(str.c_str(), str.size() * sizeof(char)));
}


