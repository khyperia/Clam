#include "client.h"
#include "server.h"
#include <GL/glut.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char** argv)
{
    glutInit(&argc, argv);

    int clientPort = 23456;
    std::vector<std::string> clients;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("port", po::value<int>(&clientPort), "Port to run client on")
        ("client", po::value(&clients), "Clients to connect to");

    po::positional_options_description positional;
    positional.add("client", -1);

    po::variables_map vm;
    po::store(
            po::command_line_parser(argc, argv).
            options(desc).
            positional(positional).
            run(),
            vm);
    po::notify(vm);

    if (clients.size() == 0)
        client(clientPort);
    else
        server(clients);
    return 0;
}
