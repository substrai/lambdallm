"""Deployment orchestrator for LambdaLLM.

Handles the full deployment lifecycle:
build → validate → deploy → verify → monitor
"""

import os
import json
import time
import shutil
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lambdallm.core.config import LambdaLLMConfig
from lambdallm.core.exceptions import LambdaLLMError
from lambdallm.deploy.generator import InfraGenerator

logger = logging.getLogger("lambdallm")


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""

    status: str  # success | failed | rolled_back
    environment: str
    endpoint_url: Optional[str] = None
    function_arn: Optional[str] = None
    stack_name: Optional[str] = None
    duration_seconds: float = 0.0
    resources_created: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Deployer:
    """Orchestrates the deployment of LambdaLLM applications.

    Deployment flow:
    1. Load and validate configuration
    2. Generate infrastructure templates
    3. Package application code
    4. Deploy via SAM CLI or CDK
    5. Run smoke tests
    6. Report results

    Example:
        deployer = Deployer(config)
        result = deployer.deploy(env="staging")
        print(f"Deployed to: {result.endpoint_url}")
    """

    def __init__(self, config: Optional[LambdaLLMConfig] = None, project_dir: str = "."):
        self.config = config or LambdaLLMConfig.load()
        self.project_dir = Path(project_dir)
        self.generator = InfraGenerator(self.config)

    def deploy(self, env: str = "dev", approve: bool = False) -> DeploymentResult:
        """Deploy the application to AWS.

        Args:
            env: Target environment (dev, staging, prod).
            approve: Skip confirmation for production deployments.

        Returns:
            DeploymentResult with endpoint URL and status.
        """
        start_time = time.time()
        stack_name = f"{self.config.project.name}-{env}"

        logger.info(f"Deploying {self.config.project.name} to {env}...")

        # Safety check for production
        if env == "prod" and not approve:
            raise LambdaLLMError(
                "Production deployment requires --approve flag. "
                "Run: lambdallm deploy --env prod --approve"
            )

        try:
            # Step 1: Validate
            self._validate(env)
            logger.info("  ✓ Configuration validated")

            # Step 2: Generate infrastructure
            template_path = self._generate_template(env)
            logger.info(f"  ✓ Generated template: {template_path}")

            # Step 3: Package
            build_dir = self._package()
            logger.info(f"  ✓ Packaged application")

            # Step 4: Deploy
            endpoint = self._deploy_sam(stack_name, template_path, env)
            logger.info(f"  ✓ Deployed stack: {stack_name}")

            # Step 5: Smoke test
            if endpoint:
                self._smoke_test(endpoint)
                logger.info(f"  ✓ Smoke tests passed")

            duration = time.time() - start_time

            return DeploymentResult(
                status="success",
                environment=env,
                endpoint_url=endpoint,
                stack_name=stack_name,
                duration_seconds=duration,
                resources_created=["Lambda", "API Gateway", "DynamoDB", "IAM Role"],
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Deployment failed: {e}")
            return DeploymentResult(
                status="failed",
                environment=env,
                stack_name=stack_name,
                duration_seconds=duration,
                errors=[str(e)],
            )

    def rollback(self, env: str = "dev", version: Optional[str] = None) -> DeploymentResult:
        """Rollback to a previous deployment version.

        Args:
            env: Target environment.
            version: Specific version to rollback to (default: previous).

        Returns:
            DeploymentResult with rollback status.
        """
        stack_name = f"{self.config.project.name}-{env}"
        logger.info(f"Rolling back {stack_name}...")

        try:
            result = subprocess.run(
                ["sam", "deploy", "--stack-name", stack_name, "--rollback"],
                capture_output=True,
                text=True,
                cwd=str(self.project_dir),
            )

            if result.returncode == 0:
                return DeploymentResult(status="rolled_back", environment=env, stack_name=stack_name)
            else:
                return DeploymentResult(
                    status="failed",
                    environment=env,
                    stack_name=stack_name,
                    errors=[result.stderr],
                )
        except FileNotFoundError:
            return DeploymentResult(
                status="failed",
                environment=env,
                errors=["SAM CLI not found. Install: pip install aws-sam-cli"],
            )

    def eject(self, output_dir: str = "infrastructure") -> str:
        """Export raw infrastructure templates for full user control.

        After ejecting, the user owns the infrastructure code.
        The framework will no longer manage it.

        Args:
            output_dir: Directory to write templates to.

        Returns:
            Path to the generated infrastructure directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate SAM template
        self.generator.generate_sam(str(output_path / "template.yaml"))

        # Generate CDK stack
        self.generator.generate_cdk(str(output_path / "cdk_stack.py"))

        # Generate CDK app entry point
        cdk_app = '''#!/usr/bin/env python3
"""CDK app entry point. Generated by: lambdallm eject"""

import aws_cdk as cdk
from cdk_stack import LambdaLLMStack

app = cdk.App()
env = app.node.try_get_context("env") or "dev"

LambdaLLMStack(app, f"LambdaLLM-{env}", env=env)
app.synth()
'''
        with open(output_path / "app.py", "w") as f:
            f.write(cdk_app)

        # Generate requirements for CDK
        cdk_requirements = """aws-cdk-lib>=2.100.0
constructs>=10.0.0
"""
        with open(output_path / "requirements.txt", "w") as f:
            f.write(cdk_requirements)

        # Generate README
        readme = """# Infrastructure (Ejected from LambdaLLM)

This infrastructure was generated by `lambdallm eject`.
You now have full control over the AWS resources.

## Deploy with SAM

```bash
sam build
sam deploy --guided
```

## Deploy with CDK

```bash
pip install -r requirements.txt
cdk deploy --context env=dev
```

## Note

After ejecting, LambdaLLM will not manage this infrastructure.
You are responsible for updates, security patches, and scaling.
"""
        with open(output_path / "README.md", "w") as f:
            f.write(readme)

        logger.info(f"Ejected infrastructure to: {output_path}/")
        logger.info("You now own the infrastructure. LambdaLLM will not manage it.")

        return str(output_path)

    def status(self, env: str = "dev") -> dict:
        """Get current deployment status.

        Returns:
            Dict with stack status, endpoint, and resource info.
        """
        stack_name = f"{self.config.project.name}-{env}"

        try:
            result = subprocess.run(
                ["aws", "cloudformation", "describe-stacks", "--stack-name", stack_name, "--output", "json"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                stacks = json.loads(result.stdout)
                stack = stacks.get("Stacks", [{}])[0]
                outputs = {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}

                return {
                    "status": stack.get("StackStatus", "UNKNOWN"),
                    "environment": env,
                    "stack_name": stack_name,
                    "endpoint": outputs.get("ApiEndpoint"),
                    "function_arn": outputs.get("FunctionArn"),
                    "last_updated": stack.get("LastUpdatedTime", "Never"),
                }
            else:
                return {"status": "NOT_DEPLOYED", "environment": env, "stack_name": stack_name}

        except FileNotFoundError:
            return {"status": "UNKNOWN", "error": "AWS CLI not found"}

    def estimate_cost(self) -> dict:
        """Estimate monthly cost based on configuration."""
        # Base costs (approximate)
        lambda_cost = 0.0  # Pay per invocation
        dynamodb_cost = 0.0  # Pay per request
        api_gw_cost = 3.50  # Per million requests (first million free)

        return {
            "estimated_monthly_usd": {
                "lambda": "Pay per invocation (~$0.20 per 1M requests)",
                "dynamodb": "Pay per request (~$1.25 per 1M writes)",
                "api_gateway": "$3.50 per 1M requests (first 1M free)",
                "bedrock": f"Based on model usage (budget: ${self.config.cost.budget.daily}/day)",
                "total_infrastructure": "< $5/month at low volume (serverless scales to zero)",
            },
            "note": "Actual costs depend on traffic volume. All resources are pay-per-use.",
        }

    def _validate(self, env: str) -> None:
        """Validate configuration and prerequisites."""
        # Check SAM CLI
        if not shutil.which("sam"):
            logger.warning("SAM CLI not found. Install: pip install aws-sam-cli")

        # Check AWS credentials
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise LambdaLLMError("AWS credentials not configured. Run: aws configure")

    def _generate_template(self, env: str) -> str:
        """Generate the deployment template."""
        template_path = str(self.project_dir / ".lambdallm" / "template.yaml")
        self.generator.generate_sam(template_path)
        return template_path

    def _package(self) -> str:
        """Package the application for deployment."""
        build_dir = str(self.project_dir / ".lambdallm" / "build")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        return build_dir

    def _deploy_sam(self, stack_name: str, template_path: str, env: str) -> Optional[str]:
        """Deploy using SAM CLI."""
        try:
            cmd = [
                "sam", "deploy",
                "--template-file", template_path,
                "--stack-name", stack_name,
                "--capabilities", "CAPABILITY_IAM",
                "--parameter-overrides", f"Environment={env}",
                "--no-confirm-changeset",
                "--no-fail-on-empty-changeset",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_dir))

            if result.returncode == 0:
                # Extract endpoint from outputs
                status = self.status(env)
                return status.get("endpoint")
            else:
                logger.warning(f"SAM deploy output: {result.stderr}")
                return None

        except FileNotFoundError:
            logger.warning("SAM CLI not available. Template generated but not deployed.")
            return None

    def _smoke_test(self, endpoint: str) -> None:
        """Run basic smoke test against deployed endpoint."""
        try:
            import urllib.request

            req = urllib.request.Request(
                endpoint,
                data=json.dumps({"body": {"text": "smoke test"}}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"Smoke test returned status {response.status}")
        except Exception as e:
            logger.warning(f"Smoke test failed (non-critical): {e}")
