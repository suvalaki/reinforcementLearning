#ifndef SERVER_H
#define SERVER_H

#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <cstdlib>
#include <iostream>
#include <optional>
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

struct WebSocketSession {
  beast::websocket::stream<beast::tcp_stream> ws;
  beast::flat_buffer buffer;

  WebSocketSession(tcp::socket &socket);

  void readMessage();
  void sendMessage(const std::string &message);
  void processMessages();
};

class WebSocketServer {
private:
  const net::ip::address address;
  unsigned short port;
  net::io_context ioc;
  tcp::acceptor acceptor;
  std::vector<std::thread> sessions = {};

public:
  WebSocketServer(const net::ip::address &address, unsigned short port);

  void acceptConnection();
  void start();
  void cleanupThreads();
  void startInNewThread();
};

std::thread launchWebSocketServerThread(net::ip::address address, unsigned short port);

} // namespace server

#endif // SERVER_H