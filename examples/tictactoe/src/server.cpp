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

void readMessage(beast::websocket::stream<beast::tcp_stream> &ws, beast::flat_buffer &buffer) {
  beast::error_code ec;
  ws.read(buffer, ec);

  if (ec == beast::websocket::error::closed) {
    ws.close(beast::websocket::close_code::normal, ec);
    throw ConnectionClosedError("Connection closed by client");
  } else if (ec) {
    throw ReadError("Read Error: " + ec.message());
  }
}

void sendMessage(beast::websocket::stream<beast::tcp_stream> &ws, const std::string &message) {
  beast::error_code write_ec;
  ws.write(boost::asio::buffer(message), write_ec);
  if (write_ec) {
    throw WriteError("Write Error: " + write_ec.message());
  }
}

void processWebSocketMessages(beast::websocket::stream<beast::tcp_stream> &ws) {
  beast::flat_buffer buffer;
  auto gameState = tictactoe::GameState();

  while (ws.is_open()) {
    buffer.consume(buffer.size());

    try {
      readMessage(ws, buffer);
    } catch (const ConnectionClosedError &e) {
      std::cerr << "Connection closed by client: " << e.what() << '\n';
      break;
    } catch (const ReadError &e) {
      std::cerr << "Read error: " << e.what() << '\n';
      break;
    }

    std::string request = beast::buffers_to_string(buffer.data());
    auto response = tictactoe::bindings::handleRequest(gameState, request);
    std::cout << response << std::endl;

    try {
      sendMessage(ws, response);
    } catch (const WriteError &e) {
      std::cerr << "Write error: " << e.what() << '\n';
      break;
    }
  }
}

void launchWebSocketSessionThread(net::io_context &ioc, tcp::socket socket) {
  std::thread{
      [&ioc](tcp::socket socket) {
        beast::websocket::stream<beast::tcp_stream> ws{std::move(socket)};
        ws.accept();
        processWebSocketMessages(ws);
      },
      std::move(socket)}
      .detach();
}

void acceptConnection(tcp::acceptor &acceptor, net::io_context &ioc) {
  tcp::socket socket{ioc};
  boost::system::error_code ec;

  acceptor.accept(socket, ec);
  if (ec) {
    throw std::runtime_error("Accept error: " + ec.message());
  }

  launchWebSocketSessionThread(ioc, std::move(socket));
}

void startWebSocketServer(const net::ip::address &address, unsigned short port) {
  try {
    net::io_context ioc;
    tcp::acceptor acceptor{ioc, {address, port}};
    std::cout << "Server is listening on port " << port << "\n";

    while (true) {
      acceptConnection(acceptor, ioc);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
  }
}

std::pair<unsigned short, std::thread> launchWebSocketServerInNewThread(const net::ip::address &address) {
  unsigned short port = getAvailablePort();
  std::thread serverThread(startWebSocketServer, address, port);
  return std::make_pair(port, std::move(serverThread));
}

}; // namespace server