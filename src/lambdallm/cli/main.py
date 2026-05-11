"""LambdaLLM CLI entry point.

Provides commands for project scaffolding, local development,
testing, and deployment.

Usage:
    lambdallm init [project-name]
    lambdallm dev
    lambdallm deploy [--env ENV]
    lambdallm test
    lambdallm cost
"""

import argparse
import sys


def cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lambdallm",
        description="LambdaLLM - Serverless-native LLM orchestration framework",
    )
    parser.add_argument("--version", action="store_true", help="Show version")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new LambdaLLM project")
    init_parser.add_argument("name", nargs="?", default="my-lambdallm-app", help="Project name")
    init_parser.add_argument("--template", choices=["basic", "chat", "agent", "rag"], default="basic", help="Starter template")

    # dev command
    dev_parser = subparsers.add_parser("dev", help="Start local development server")
    dev_parser.add_argument("--port", type=int, default=3000, help="Port number")
    dev_parser.add_argument("--handler", help="Specific handler to serve")

    # deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to AWS")
    deploy_parser.add_argument("--env", default="dev", help="Target environment")
    deploy_parser.add_argument("--approve", action="store_true", help="Skip confirmation for prod")

    # test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--golden", action="store_true", help="Run golden dataset tests")
    test_parser.add_argument("--cost-estimate", action="store_true", help="Estimate test costs")

    # cost command
    cost_parser = subparsers.add_parser("cost", help="Show cost summary")
    cost_parser.add_argument("--forecast", action="store_true", help="Forecast monthly spend")

    # invoke command
    invoke_parser = subparsers.add_parser("invoke", help="Invoke a handler locally")
    invoke_parser.add_argument("handler", help="Handler name")
    invoke_parser.add_argument("--data", help="JSON input data")
    invoke_parser.add_argument("--file", help="Input data from file")

    # eject command
    subparsers.add_parser("eject", help="Export raw CDK/SAM templates")

    args = parser.parse_args()

    if args.version:
        from lambdallm import __version__
        print(f"lambdallm {__version__}")
        return

    if args.command is None:
        parser.print_help()
        return

    # Dispatch to command handlers
    commands = {
        "init": cmd_init,
        "dev": cmd_dev,
        "deploy": cmd_deploy,
        "test": cmd_test,
        "cost": cmd_cost,
        "invoke": cmd_invoke,
        "eject": cmd_eject,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


def cmd_init(args):
    """Initialize a new LambdaLLM project."""
    from lambdallm.cli.init import init_project
    init_project(args.name, args.template)


def cmd_dev(args):
    """Start local development server."""
    from lambdallm.cli.dev import start_dev_server
    start_dev_server(port=args.port, handler=args.handler)


def cmd_deploy(args):
    """Deploy to AWS."""
    print(f"Deploying to environment: {args.env}")
    print("Deploy command coming in Phase 5. Use SAM/CDK directly for now.")
    print("Run: lambdallm eject  to get the infrastructure templates.")


def cmd_test(args):
    """Run tests."""
    import subprocess
    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    if args.golden:
        cmd.extend(["--golden"])
    subprocess.run(cmd)


def cmd_cost(args):
    """Show cost summary."""
    print("Cost tracking dashboard coming in Phase 4.")
    print("Current request costs are logged to CloudWatch.")


def cmd_invoke(args):
    """Invoke a handler locally."""
    from lambdallm.cli.dev import invoke_handler
    invoke_handler(args.handler, args.data, args.file)


def cmd_eject(args):
    """Export raw infrastructure templates."""
    print("Eject command coming in Phase 5.")
    print("This will export CDK/SAM templates for full infrastructure control.")


if __name__ == "__main__":
    cli()
