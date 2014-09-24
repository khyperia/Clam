#include "client.h"
#include "server.h"

int main(int argc, char** argv)
{
    if (argv[1] && argv[1][0] == '-' ? argc <= 2 : argc <= 1)
        client(argc, argv);
    else
        server(argc, argv);
    return 0;
}
