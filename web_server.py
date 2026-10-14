# -*- coding: utf-8 -*-
"""
HW2 Bike Sharing Demand Prediction - Web Dashboard Server
=========================================================
Lightweight local HTTP server serving the interactive web dashboard,
results visualizations, reports, and submission downloads.
"""

import os
import sys
import http.server
import socketserver

# Ensure UTF-8 console output for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PORT = 8080

class ProjectHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Redirect root and web static files to web/ directory
        if self.path in ('/', '/index.html'):
            self.path = '/web/index.html'
        elif self.path in ('/style.css', '/app.js'):
            self.path = '/web' + self.path
        return super().do_GET()

    def end_headers(self):
        # Add CORS and cache-control headers for development
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def run_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ProjectHTTPRequestHandler) as httpd:
        print("\n========================================================")
        print(f"HW2 共享單車需求預測系統 Web Dashboard 已啟動！")
        print(f"請開啟瀏覽器訪問: http://localhost:{PORT}")
        print(f"或訪問本機 IP: http://127.0.0.1:{PORT}")
        print("========================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n正在關閉 Web 伺服器...")
            httpd.server_close()

if __name__ == '__main__':
    run_server()
