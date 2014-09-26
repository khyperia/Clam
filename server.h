#include <string>
#include <vector>
#include <memory>
#include <boost/asio.hpp>

void server(std::string kernelFile, std::vector<std::string> clients);
bool iskeydown(unsigned char key);
std::shared_ptr<std::vector<std::shared_ptr<boost::asio::ip::tcp::socket>>> getSocks();
