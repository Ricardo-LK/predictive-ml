import argparse


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: ``predictive-ml {train,serve}``."""
    parser = argparse.ArgumentParser(
        prog="predictive-ml",
        description="Predictive maintenance model training and inference API.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "train",
        help="Download AI4I 2020, compare models, pick a threshold, "
        "evaluate on the held-out test set and save the model (needs internet).",
    )

    serve = sub.add_parser("serve", help="Start the FastAPI inference server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "train":
        from predictive_ml.train import main as train_main

        train_main()
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("predictive_ml.api:app", host=args.host, port=args.port)
