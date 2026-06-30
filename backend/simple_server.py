from http.server import HTTPServer, BaseHTTPRequestHandler

class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Hello World")


print("Creating server...")

server = HTTPServer(("127.0.0.1", 9000), MyHandler)

print("Server Started!")

server.serve_forever()