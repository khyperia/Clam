import http.server
import socketserver

PORT = 8080

class HttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        '.js': 'text/javascript',
    }

httpd = socketserver.TCPServer(("localhost", PORT), HttpRequestHandler)

try:
    print(f"serving at http://localhost:{PORT}")
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
