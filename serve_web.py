import argparse
import uvicorn
from lerobot_vision import web_interface


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the web interface")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        web_interface.app, host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
