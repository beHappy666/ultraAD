"""Web Studio command implementation."""

import os
import sys
import click
import uvicorn
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.command('studio')
@click.option('--host', '-h', default='0.0.0.0', help='Host to bind to')
@click.option('--port', '-p', type=int, default=8080, help='Port to listen on')
@click.option('--reload', '-r', is_flag=True, help='Enable auto-reload')
@click.option('--no-browser', is_flag=True, help='Do not open browser')
def studio_cmd(host, port, reload, no_browser):
    """Launch ultraAD Studio - Web-based debugging and visualization.

    The Studio provides an interactive web interface for:
    - Real-time training monitoring
    - 3D scene visualization
    - BEV feature inspection
    - Model debugging and profiling

    Examples:
        ultraad studio                    # Launch on default port 8080
        ultraad studio -p 8888            # Use custom port
        ultraad studio --no-browser       # Don't open browser
    """
    console.print(Panel.fit(
        "[bold cyan]ultraAD Studio[/] - Web-based Development Environment\n"
        f"Starting server at http://{host}:{port}",
        title="Studio"
    ))

    # Check if web module exists
    web_dir = Path(__file__).parent.parent.parent / 'web'
    if not web_dir.exists():
        console.print("[yellow]Warning: web/ directory not found. Creating minimal setup...")
        _create_minimal_web_setup(web_dir)

    # Open browser
    if not no_browser:
        import webbrowser
        url = f"http://localhost:{port}"
        console.print(f"[green]Opening browser at {url}")
        webbrowser.open(url)

    # Start server
    try:
        # Check for FastAPI app
        backend_dir = web_dir / 'backend'
        if (backend_dir / 'main.py').exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("main", backend_dir / 'main.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            app = module.app
        else:
            # Create minimal FastAPI app
            from fastapi import FastAPI
            from fastapi.staticfiles import StaticFiles
            from fastapi.responses import HTMLResponse

            app = FastAPI(title="ultraAD Studio")

            @app.get("/", response_class=HTMLResponse)
            async def root():
                return """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>ultraAD Studio</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        h1 { color: #00bcd4; }
                        .card { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 8px; }
                    </style>
                </head>
                <body>
                    <h1>ultraAD Studio</h1>
                    <p>Welcome to the ultraAD Web Interface!</p>

                    <div class="card">
                        <h3>Getting Started</h3>
                        <p>This is a minimal setup. To enable full features:</p>
                        <ul>
                            <li>Install frontend dependencies: <code>cd web/frontend && npm install</code></li>
                            <li>Build frontend: <code>npm run build</code></li>
                        </ul>
                    </div>

                    <div class="card">
                        <h3>API Status</h3>
                        <p>Backend API is running at <a href="/docs">/docs</a></p>
                    </div>
                </body>
                </html>
                """

            # Add API routes
            @app.get("/api/status")
            async def status():
                return {"status": "ok", "version": "0.2.0"}

        # Run server
        uvicorn.run(app, host=host, port=port, reload=reload)

    except Exception as e:
        console.print(f"[red]Error starting server: {e}")
        sys.exit(1)


def _create_minimal_web_setup(web_dir):
    """Create minimal web setup if it doesn't exist."""
    web_dir = Path(web_dir)

    # Create directories
    (web_dir / 'backend').mkdir(parents=True, exist_ok=True)
    (web_dir / 'frontend').mkdir(parents=True, exist_ok=True)

    # Create minimal backend main.py
    backend_main = web_dir / 'backend' / 'main.py'
    backend_main.write_text('''
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ultraAD Studio API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "ultraAD Studio API", "version": "0.2.0"}

@app.get("/api/status")
async def status():
    return {"status": "ok"}
''')

    # Create minimal frontend package.json
    package_json = web_dir / 'frontend' / 'package.json'
    package_json.write_text('''
{
  "name": "ultraad-studio-frontend",
  "version": "0.2.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "scripts": {
    "build": "echo 'Build not implemented yet'",
    "dev": "echo 'Dev server not implemented yet'"
  }
}
''')

    console.print(f"[green]Created minimal web setup at {web_dir}")