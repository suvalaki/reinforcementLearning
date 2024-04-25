#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <thread>
#include <cstdlib>
#include <iostream>
#include <string>

#include "bindings.hpp"

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace net = boost::asio;             // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>

int main(int argc, char* argv[])
{
    try
    {
        // Check command line arguments.
        if (argc != 3)
        {
            std::cerr << "Usage: websocket-server <address> <port>\n";
            return EXIT_FAILURE;
        }
        auto const address = net::ip::make_address(argv[1]);
        auto const port = static_cast<unsigned short>(std::atoi(argv[2]));

        // The io_context is required for all I/O
        net::io_context ioc;

        // The acceptor receives incoming connections
        tcp::acceptor acceptor{ioc, {address, port}};
        for(;;)
        {
            // This will receive the new connection
            tcp::socket socket{ioc};

            // Block until we get a connection
            acceptor.accept(socket);

            // Launch the session, transferring ownership of the socket
            std::thread{[&ioc](tcp::socket socket)
            {
                beast::websocket::stream<beast::tcp_stream> ws{std::move(socket)};
                ws.accept();

                beast::flat_buffer buffer;

                auto gameState = tictactoe::GameState();

                // Keep reading messages until connection is closed by the client
                while(ws.is_open())
                {
                    // Clear the buffer
                    buffer.consume(buffer.size());

                    // Read a message
                    beast::error_code ec;
                    ws.read(buffer, ec);

                    // If a close frame is received, close the connection
                    if(ec == beast::websocket::error::closed)
                    {
                        ws.close(beast::websocket::close_code::normal, ec);
                        break;
                    }
                    else if(ec)
                    {
                        std::cerr << "Error: " << ec.message() << '\n';
                        break;
                    }

                    // Print the message
                    //std::cout << beast::make_printable(buffer.data()) << std::endl;
                    std::string request = beast::buffers_to_string(buffer.data());
                    auto ret = tictactoe::bindings::handleRequest(gameState, request);
                    std::cout << ret << std::endl;

                    // Send the payload back to the client
                    beast::error_code write_ec;
                    ws.write(boost::asio::buffer(ret), write_ec);
                    if(write_ec)
                    {
                        std::cerr << "Write Error: " << write_ec.message() << '\n';
                        break;
                    }

                }

            }, std::move(socket)}.detach();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}