from http.server import HTTPServer, BaseHTTPRequestHandler

class TestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Hello")

server = HTTPServer(("127.0.0.1",9000), TestHandler)

print("Server Running")

server.serve_forever()