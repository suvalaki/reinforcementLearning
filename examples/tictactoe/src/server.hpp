#ifndef SERVER_H
#define SERVER_H

#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

#include "bindings.hpp"

namespace server {

namespace beast = boost::beast;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

class ConnectionClosedError : public std::runtime_error {
public:
  ConnectionClosedError(const std::string &message);
};

class ReadError : public std::runtime_error {
public:
  ReadError(const std::string &message);
};

class WriteError : public std::runtime_error {
public:
  WriteError(const std::string &message);
};

unsigned short getAvailablePort();

void readMessage(beast::websocket::stream<beast::tcp_stream> &ws, beast::flat_buffer &buffer);

void sendMessage(beast::websocket::stream<beast::tcp_stream> &ws, const std::string &message);

void processWebSocketMessages(beast::websocket::stream<beast::tcp_stream> &ws);

void launchWebSocketSessionThread(net::io_context &ioc, tcp::socket socket);

void acceptConnection(tcp::acceptor &acceptor, net::io_context &ioc);

void startWebSocketServer(const net::ip::address &address, unsigned short port);

std::pair<unsigned short, std::thread> launchWebSocketServerInNewThread(const net::ip::address &address);

} // namespace server

#endif // SERVER_H