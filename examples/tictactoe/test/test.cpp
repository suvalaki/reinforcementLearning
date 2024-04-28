#include <saucer/smartview.hpp>

int main() 
{
  saucer::smartview webview;
  
  webview.set_size(500, 600);
  webview.set_title("Hello World!");

    smartview.set_url("index.html");
  webview.serve("index.html");

  webview.show();
  webview.run();

  return 0;
}