"""LambdaLLM CLI entry point.

Provides commands for project scaffolding, local development,
testing, deployment, and monitoring.

Usage:
    lambdallm init [project-name]
    lambdallm dev
    lambdallm test
    lambdallm deploy [--env ENV]
    lambdallm status [--env ENV]
    lambdallm rollback [--env ENV]
    lambdallm cost [--forecast]
    lambdallm eject
"""

import argparse
import sys
import json


def cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lambdallm",
        description="LambdaLLM - Serverless-native LLM orchestration framework",
    )
    parser.add_argument("--version", action="store_true", help="Show version")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument("name", nargs="?", default="my-lambdallm-app", help="Project name")
    init_parser.add_argument("--template", choices=["basic", "chat", "agent", "rag"], default="basic")

    # dev
    dev_parser = subparsers.add_parser("dev", help="Start local development server")
    dev_parser.add_argument("--port", type=int, default=3000)
    dev_parser.add_argument("--handler", help="Handler module path")

    # deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to AWS")
    deploy_parser.add_argument("--env", default="dev", help="Target environment")
    deploy_parser.add_argument("--approve", action="store_true", help="Skip prod confirmation")

    # status
    status_parser = subparsers.add_parser("status", help="Show deployment status")
    status_parser.add_argument("--env", default="dev")

    # rollback
    rollback_parser = subparsers.add_parser("rollback", help="Rollback deployment")
    rollback_parser.add_argument("--env", default="dev")
    rollback_parser.add_argument("--to", dest="version", help="Specific version")

    # test
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--golden", action="store_true", help="Run golden dataset tests")
    test_parser.add_argument("--cost-estimate", action="store_true")

    # cost
    cost_parser = subparsers.add_parser("cost", help="Show cost information")
    cost_parser.add_argument("--forecast", action="store_true")
    cost_parser.add_argument("--report", choices=["daily", "monthly"], default="daily")

    # invoke
    invoke_parser = subparsers.add_parser("invoke", help="Invoke handler locally")
    invoke_parser.add_argument("handler", help="Handler name")
    invoke_parser.add_argument("--data", help="JSON input data")
    invoke_parser.add_argument("--file", help="Input from file")

    # eject
    eject_parser = subparsers.add_parser("eject", help="Export raw infrastructure templates")
    eject_parser.add_argument("--output", default="infrastructure", help="Output directory")

    # logs
    logs_parser = subparsers.add_parser("logs", help="Tail CloudWatch logs")
    logs_parser.add_argument("--env", default="dev")
    logs_parser.add_argument("--follow", "-f", action="store_true")

    # metrics
    subparsers.add_parser("metrics", help="Show key metrics")

    args = parser.parse_args()

    if args.version:
        from lambdallm import __version__
        print(f"lambdallm {__version__}")
        return

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "init": cmd_init,
        "dev": cmd_dev,
        "deploy": cmd_deploy,
        "status": cmd_status,
        "rollback": cmd_rollback,
        "test": cmd_test,
        "cost": cmd_cost,
        "invoke": cmd_invoke,
        "eject": cmd_eject,
        "logs": cmd_logs,
        "metrics": cmd_metrics,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


def cmd_init(args):
    """Initialize a new project."""
    from lambdallm.cli.init import init_project
    init_project(args.name, args.template)


def cmd_dev(args):
    """Start local development server."""
    from lambdallm.cli.dev import start_dev_server
    start_dev_server(port=args.port, handler=args.handler)


def cmd_deploy(args):
    """Deploy to AWS."""
    from lambdallm.deploy.deployer import Deployer
    from lambdallm.core.config import LambdaLLMConfig

    config = LambdaLLMConfig.load()
    deployer = Deployer(config)

    print(f"Deploying {config.project.name} to {args.env}...")
    print()

    result = deployer.deploy(env=args.env, approve=args.approve)

    if result.status == "success":
        print(f"  ✓ Deployment successful!")
        print(f"  Environment: {result.environment}")
        if result.endpoint_url:
            print(f"  Endpoint: {result.endpoint_url}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Resources: {', '.join(result.resources_created)}")
    else:
        print(f"  ✗ Deployment failed")
        for error in result.errors:
            print(f"    Error: {error}")
        sys.exit(1)


def cmd_status(args):
    """Show deployment status."""
    from lambdallm.deploy.deployer import Deployer
    from lambdallm.core.config import LambdaLLMConfig

    config = LambdaLLMConfig.load()
    deployer = Deployer(config)

    status = deployer.status(env=args.env)
    print(f"Stack: {status.get('stack_name', 'N/A')}")
    print(f"Status: {status.get('status', 'UNKNOWN')}")
    print(f"Environment: {status.get('environment', args.env)}")
    if status.get("endpoint"):
        print(f"Endpoint: {status['endpoint']}")
    if status.get("last_updated"):
        print(f"Last Updated: {status['last_updated']}")


def cmd_rollback(args):
    """Rollback deployment."""
    from lambdallm.deploy.deployer import Deployer
    from lambdallm.core.config import LambdaLLMConfig

    config = LambdaLLMConfig.load()
    deployer = Deployer(config)

    print(f"Rolling back {args.env}...")
    result = deployer.rollback(env=args.env, version=args.version)

    if result.status == "rolled_back":
        print("  ✓ Rollback successful")
    else:
        print(f"  ✗ Rollback failed: {result.errors}")
        sys.exit(1)


def cmd_test(args):
    """Run tests."""
    import subprocess

    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    if args.cost_estimate:
        print("Estimated test cost: ~$0.01 (mock providers used by default)")
        print()
    subprocess.run(cmd)


def cmd_cost(args):
    """Show cost information."""
    from lambdallm.deploy.deployer import Deployer
    from lambdallm.core.config import LambdaLLMConfig

    config = LambdaLLMConfig.load()
    deployer = Deployer(config)

    if args.forecast:
        from lambdallm.observability.cost_tracker import CostTracker
        tracker = CostTracker(
            daily_budget=config.cost.budget.daily,
            monthly_budget=config.cost.budget.monthly,
        )
        forecast = tracker.forecast_monthly()
        print("Monthly Forecast:")
        print(f"  Current spend: ${forecast['current_monthly_spend']:.4f}")
        print(f"  Daily average: ${forecast['daily_average']:.4f}")
        print(f"  Projected: ${forecast['projected_monthly']:.2f}")
        print(f"  Budget: ${forecast['monthly_budget']:.2f}")
        print(f"  On track: {'✓' if forecast['on_track'] else '✗ OVER BUDGET'}")
    else:
        estimate = deployer.estimate_cost()
        print("Infrastructure Cost Estimate:")
        for service, cost in estimate["estimated_monthly_usd"].items():
            print(f"  {service}: {cost}")
        print(f"\n  Note: {estimate['note']}")


def cmd_invoke(args):
    """Invoke handler locally."""
    from lambdallm.cli.dev import invoke_handler
    invoke_handler(args.handler, args.data, args.file)


def cmd_eject(args):
    """Export infrastructure templates."""
    from lambdallm.deploy.deployer import Deployer
    from lambdallm.core.config import LambdaLLMConfig

    config = LambdaLLMConfig.load()
    deployer = Deployer(config)

    output = deployer.eject(output_dir=args.output)
    print(f"Infrastructure ejected to: {output}/")
    print()
    print("Files generated:")
    print(f"  {output}/template.yaml  (SAM template)")
    print(f"  {output}/cdk_stack.py   (CDK stack)")
    print(f"  {output}/app.py         (CDK entry point)")
    print(f"  {output}/requirements.txt")
    print(f"  {output}/README.md")
    print()
    print("You now own the infrastructure. LambdaLLM will not manage it.")
    print("Deploy with: cd infrastructure && sam build && sam deploy --guided")


def cmd_logs(args):
    """Tail CloudWatch logs."""
    import subprocess

    stack_name = f"lambdallm-{args.env}"
    log_group = f"/aws/lambda/{stack_name}"

    cmd = ["aws", "logs", "tail", log_group, "--format", "short"]
    if args.follow:
        cmd.append("--follow")

    print(f"Tailing logs for: {log_group}")
    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        print("AWS CLI not found. Install: https://aws.amazon.com/cli/")
    except KeyboardInterrupt:
        print("\nStopped.")


def cmd_metrics(args):
    """Show key metrics."""
    print("LambdaLLM Metrics Dashboard")
    print("=" * 40)
    print()
    print("View in CloudWatch:")
    print("  https://console.aws.amazon.com/cloudwatch/home#metricsV2:namespace=LambdaLLM")
    print()
    print("Key metrics tracked:")
    print("  • handler.latency (ms)")
    print("  • handler.errors (count)")
    print("  • model.invocations (count)")
    print("  • model.cost_usd (dollars)")
    print("  • model.tokens_in / tokens_out")
    print("  • agent.iterations / tool_calls")
    print("  • budget.utilization_percent")


if __name__ == "__main__":
    cli()
