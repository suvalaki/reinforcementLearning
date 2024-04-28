#include <saucer/smartview.hpp>

#include <fstream>
#include <sstream>
#include <string>

#include "server.hpp"

#define LOCALHOST_IP "0.0.0.0"
#define LOOPBACK_IP "127.0.0.1"

std::string readFileIntoString(const std::string& filePath) {
    std::ifstream fileStream(filePath);
    std::stringstream stringStream;

    if (!fileStream) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    stringStream << fileStream.rdbuf();
    return stringStream.str();
}

saucer::embedded_file convertStringToEmbeddedFile(const std::string& content, const std::string& mime = "text/html") {
    return saucer::embedded_file{
        mime,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(content.data()), content.size())
    };
}

int main() 
{

  // Start the game
  auto [port, server] = server::launchWebSocketServerInNewThread(boost::asio::ip::make_address(LOCALHOST_IP));

  saucer::smartview webview;
  webview.set_size(500, 600);


    // Requires you to run the executable with these files handy
  std::string indexHtmlContents = readFileIntoString("index.html");
  webview.embed({{"index.html", convertStringToEmbeddedFile(indexHtmlContents, "text/html") }});

  std::string indexJsContents = readFileIntoString("client.js");
  webview.embed({{"client.js", convertStringToEmbeddedFile(indexJsContents, "text/javascript") }});

  webview.expose("get_port", [&](){return port;}, true);

  webview.serve("index.html");

  //webview.set_dev_tools(true);
  webview.show();
  webview.run();

  return 0;
}