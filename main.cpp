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
    std::string kernel;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("port", po::value<int>(&clientPort), "Port to run client on")
        ("client", po::value(&clients), "Clients to connect to")
        ("kernel", po::value(&kernel), "Kernel file to use");

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

    try
    {
        if (clients.size() == 0)
            client(clientPort);
        else
        {
            if (kernel.empty())
            {
                puts("\"--kernel filename.cl\" not provided");
            }
            server(kernel, clients);
        }
    }
    catch (std::exception const& ex)
    {
        puts("Unhandled exception:");
        puts(ex.what());
        return -1;
    }
    return 0;
}
