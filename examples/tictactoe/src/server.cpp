#include "server.hpp"

namespace server {

ConnectionClosedError::ConnectionClosedError(const std::string &message) : std::runtime_error(message) {}

ReadError::ReadError(const std::string &message) : std::runtime_error(message) {}

WriteError::WriteError(const std::string &message) : std::runtime_error(message) {}

unsigned short getAvailablePort() {
  boost::asio::io_context ioc;
  boost::asio::ip::tcp::acceptor acceptor(ioc);
  acceptor.open(boost::asio::ip::tcp::v4());
  acceptor.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
  unsigned short port = acceptor.local_endpoint().port();
  return port;
}

WebSocketSession::WebSocketSession(
    tcp::socket &socket, const tictactoe::bindings::game_control_factory_t &gameControlFactory)
    : ws(beast::websocket::stream<beast::tcp_stream>(std::move(socket))), buffer(beast::flat_buffer{}),
      gameControl(gameControlFactory()) {}

void WebSocketSession::readMessage() {
  beast::error_code ec;
  ws.read(buffer, ec);

  if (ec == beast::websocket::error::closed) {
    ws.close(beast::websocket::close_code::normal, ec);
    throw ConnectionClosedError("Connection closed by client");
  } else if (ec) {
    throw ReadError("Read Error: " + ec.message());
  }
}

void WebSocketSession::sendMessage(const std::string &message) {
  beast::error_code write_ec;
  ws.write(boost::asio::buffer(message), write_ec);
  if (write_ec) {
    throw WriteError("Write Error: " + write_ec.message());
  }
}

void WebSocketSession::processMessages() {

  while (ws.is_open()) {
    buffer.consume(buffer.size());

    try {
      readMessage();
    } catch (const ConnectionClosedError &e) {
      std::cerr << "Connection closed by client: " << e.what() << '\n';
      break;
    } catch (const ReadError &e) {
      std::cerr << "Read error: " << e.what() << '\n';
      break;
    }

    std::string request = beast::buffers_to_string(buffer.data());
    auto response = gameControl->handleRequest(request);
    std::cout << response << std::endl;

    try {
      sendMessage(response);
    } catch (const WriteError &e) {
      std::cerr << "Write error: " << e.what() << '\n';
      break;
    }
  }
}

WebSocketServer::WebSocketServer(
    const net::ip::address &address,
    unsigned short port,
    const tictactoe::bindings::game_control_factory_t &gameControlFactory)
    : address(address), port(port), ioc(net::io_context()), acceptor(ioc, {address, port}),
      gameControlFactory(gameControlFactory) {}

void WebSocketServer::acceptConnection() {
  tcp::socket socket{ioc};
  boost::system::error_code ec;

  acceptor.accept(socket, ec);
  if (ec) {
    throw std::runtime_error("Accept error: " + ec.message());
  }

  // sessions.emplace_back(WebSocketSession(socket));
  auto &item = sessions.emplace_back(std::thread(
      [this](tcp::socket socket) {
        auto session = WebSocketSession(socket, gameControlFactory);
        session.ws.accept();
        session.processMessages();
      },
      std::move(socket)));
  item.detach();
}

void WebSocketServer::start() {
  try {
    std::cout << "Server is listening on port " << port << "\n";

    while (true) {
      acceptConnection();
      cleanupThreads();
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
  }
}

void WebSocketServer::cleanupThreads() {
  sessions.erase(
      std::remove_if(
          sessions.begin(),
          sessions.end(),
          [](std::thread &s) {
            if (s.joinable()) {
              s.join();
              return true;
            }
            return false;
          }),
      sessions.end());
}

void WebSocketServer::startInNewThread() {
  std::thread([this] { start(); }).detach();
}

std::thread launchWebSocketServerThread(
    net::ip::address address,
    unsigned short port,
    const tictactoe::bindings::game_control_factory_t &gameControlFactory) {
  auto serverThread = std::thread([address, port, gameControlFactory] {
    auto server = WebSocketServer(address, port, gameControlFactory);
    server.start();
  });
  serverThread.detach();
  return serverThread;
}

}; // namespace server