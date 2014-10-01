#include "echo.h"

#include "socket.h"
#include <chrono>
#include <map>

void runEcho(std::shared_ptr<CSocket> copyFrom,
        std::shared_ptr<CSocket> copyTo,
        std::shared_ptr<int> numThreads)
{
    try
    {
        uint8_t byte;
        auto lastUpdate = std::chrono::steady_clock::now();
        while (true)
        {
            while (copyFrom->RecvByte(&byte))
            {
                copyTo->SendByte(byte);
                lastUpdate = std::chrono::steady_clock::now();
            }
            while (copyTo->RecvByte(&byte))
            {
                copyFrom->SendByte(byte);
                lastUpdate = std::chrono::steady_clock::now();
            }
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - lastUpdate).count() > 3)
                break;
        }
        puts("Echo thread timeout: killing");
    }
    catch (std::exception const& ex)
    {
        puts("Echo thread died:");
        puts(ex.what());
    }
    (*numThreads)--;
}

// TODO: Find a good clean way to stop this without pkill
void echo(const char* incomingPort, std::vector<std::string> outgoingIps)
{
    auto host = std::make_shared<CSocket>(incomingPort);
    std::map<std::string, std::pair<std::thread, std::shared_ptr<int>>> connections;
    while (true)
    {
        auto newSocket = host->Accept();

        try
        {
            std::string client;
            for (auto const& outgoing : outgoingIps)
            {
                auto found = connections.find(outgoing);
                if (found != connections.end())
                {
                    if (*found->second.second == 0)
                    {
                        client = outgoing;
                    }
                }
                else
                {
                    client = outgoing;
                }
            }

            if (client.empty())
            {
                puts("Got connection when no more were left");
                newSocket = nullptr;
                continue;
            }

            auto otherSocket = std::make_shared<CSocket>(client);
            puts(("Echo server connected to " + client).c_str());
            auto numThreads = std::make_shared<int>(1);
            auto foundt = connections.find(client);
            if (foundt != connections.end() && foundt->second.first.joinable())
                foundt->second.first.join();
            connections[client] = std::make_pair(
                    std::thread([otherSocket, newSocket, numThreads]{
                        runEcho(otherSocket, newSocket, numThreads);
                        }),
                    numThreads);
        }
        catch (std::exception const& ex)
        {
            puts("Echo server: Something went wrong connecting to client:");
            puts(ex.what());
        }
    }
}
