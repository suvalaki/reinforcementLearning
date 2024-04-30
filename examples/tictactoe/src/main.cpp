#include <saucer/smartview.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "server.hpp"
#include "tracing.hpp"

#define LOCALHOST_IP "0.0.0.0"
#define LOOPBACK_IP "127.0.0.1"

std::string readFileIntoString(const std::string &filePath) {
  std::ifstream fileStream(filePath);
  std::stringstream stringStream;

  if (!fileStream) {
    throw std::runtime_error("Could not open file: " + filePath);
  }

  stringStream << fileStream.rdbuf();
  return stringStream.str();
}

saucer::embedded_file convertStringToEmbeddedFile(const std::string &content, const std::string &mime = "text/html") {
  return saucer::embedded_file{
      mime, std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t *>(content.data()), content.size())};
}

int main() {

  // Create database

  // Start the game
  const auto logGameControlFactory = []() {
    std::shared_ptr<tictactoe::connection_t> my_conn;
    auto created_conn = tictactoe::connection_t("test.db", 0, 0, nullptr);
    my_conn = std::make_shared<tictactoe::connection_t>(std::move(created_conn));
    auto logger = tictactoe::DatabaseLogger(my_conn);
    return std::make_unique<tictactoe::bindings::LoggedGameControl>(logger);
  };
  auto port = server::getAvailablePort();
  auto server =
      server::launchWebSocketServerThread(boost::asio::ip::make_address(LOCALHOST_IP), port, logGameControlFactory);

  saucer::smartview webview;
  webview.set_size(500, 600);

  // Requires you to run the executable with these files handy
  std::string indexHtmlContents = readFileIntoString("index.html");
  webview.embed({{"index.html", convertStringToEmbeddedFile(indexHtmlContents, "text/html")}});

  std::string indexJsContents = readFileIntoString("client.js");
  webview.embed({{"client.js", convertStringToEmbeddedFile(indexJsContents, "text/javascript")}});

  webview.expose(
      "get_port", [&]() { return port; }, true);

  webview.serve("index.html");

  // webview.set_dev_tools(true);
  webview.show();
  webview.run();

  return 0;
}