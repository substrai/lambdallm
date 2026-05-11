"""Infrastructure generator for LambdaLLM.

Generates SAM/CDK templates from lambdallm.yaml configuration.
Users never write CloudFormation — the framework generates it.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lambdallm.core.config import LambdaLLMConfig

logger = logging.getLogger("lambdallm")


@dataclass
class InfraResource:
    """A single infrastructure resource to be provisioned."""

    type: str  # lambda | api_gateway | dynamodb | cloudwatch | iam
    logical_name: str
    properties: dict = field(default_factory=dict)


class InfraGenerator:
    """Generates AWS infrastructure templates from LambdaLLM config.

    Supports two output formats:
    - SAM (template.yaml) — lightweight, single-file
    - CDK (cdk_stack.py) — full programmatic control

    Example:
        generator = InfraGenerator(config)
        generator.generate_sam("./infrastructure/template.yaml")
        generator.generate_cdk("./infrastructure/cdk_stack.py")
    """

    def __init__(self, config: Optional[LambdaLLMConfig] = None):
        self.config = config or LambdaLLMConfig.load()
        self._resources: list[InfraResource] = []

    def generate_sam(self, output_path: str = "template.yaml") -> str:
        """Generate a SAM template from configuration.

        Args:
            output_path: Where to write the template.

        Returns:
            The generated template content.
        """
        self._discover_resources()
        template = self._build_sam_template()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(template)

        logger.info(f"Generated SAM template: {output_path} ({len(self._resources)} resources)")
        return template

    def generate_cdk(self, output_path: str = "infrastructure/cdk_stack.py") -> str:
        """Generate a CDK stack from configuration.

        Args:
            output_path: Where to write the CDK stack.

        Returns:
            The generated CDK code.
        """
        self._discover_resources()
        cdk_code = self._build_cdk_stack()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(cdk_code)

        logger.info(f"Generated CDK stack: {output_path}")
        return cdk_code

    def _discover_resources(self) -> None:
        """Discover required resources from config and handlers."""
        self._resources = []

        # Lambda function for each handler
        self._resources.append(InfraResource(
            type="lambda",
            logical_name="LambdaLLMHandler",
            properties={
                "runtime": self.config.project.runtime,
                "timeout": self.config.defaults.timeout,
                "memory": self.config.defaults.memory,
                "handler": "handlers.main.lambda_handler",
            },
        ))

        # API Gateway
        self._resources.append(InfraResource(
            type="api_gateway",
            logical_name="LambdaLLMApi",
            properties={"stage": "prod", "cors": True},
        ))

        # DynamoDB for sessions (if state is configured)
        if self.config.defaults.state.provider == "dynamodb":
            self._resources.append(InfraResource(
                type="dynamodb",
                logical_name="SessionsTable",
                properties={
                    "table_name": f"{self.config.defaults.state.table_prefix}sessions",
                    "partition_key": "session_id",
                    "ttl_attribute": "expires_at",
                    "billing_mode": "PAY_PER_REQUEST",
                },
            ))

            # Cost tracking table
            self._resources.append(InfraResource(
                type="dynamodb",
                logical_name="CostsTable",
                properties={
                    "table_name": f"{self.config.defaults.state.table_prefix}costs",
                    "partition_key": "date_key",
                    "sort_key": "timestamp",
                    "billing_mode": "PAY_PER_REQUEST",
                },
            ))

        # IAM Role
        self._resources.append(InfraResource(
            type="iam",
            logical_name="LambdaExecutionRole",
            properties={
                "services": ["bedrock", "dynamodb", "cloudwatch", "xray"],
            },
        ))

        # CloudWatch Dashboard
        self._resources.append(InfraResource(
            type="cloudwatch",
            logical_name="LambdaLLMDashboard",
            properties={"metrics_namespace": "LambdaLLM"},
        ))

    def _build_sam_template(self) -> str:
        """Build SAM template YAML."""
        project_name = self.config.project.name
        runtime = self.config.defaults.state.table_prefix

        template = f"""AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: '{project_name} - Generated by LambdaLLM'

Globals:
  Function:
    Timeout: {self.config.defaults.timeout}
    MemorySize: {self.config.defaults.memory}
    Runtime: {self.config.project.runtime}
    Tracing: Active
    Environment:
      Variables:
        LAMBDALLM_ENV: !Ref Environment
        LAMBDALLM_TABLE_PREFIX: {runtime}

Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]

Resources:
  LambdaLLMFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: .
      Handler: handlers.main.lambda_handler
      Description: 'LambdaLLM handler'
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref SessionsTable
        - DynamoDBCrudPolicy:
            TableName: !Ref CostsTable
        - Statement:
            - Effect: Allow
              Action:
                - bedrock:InvokeModel
                - bedrock:InvokeModelWithResponseStream
              Resource: '*'
      Events:
        Api:
          Type: Api
          Properties:
            Path: /
            Method: POST
        ApiAny:
          Type: Api
          Properties:
            Path: /{{proxy+}}
            Method: ANY

  SessionsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub '{runtime}sessions-${{Environment}}'
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: session_id
          AttributeType: S
      KeySchema:
        - AttributeName: session_id
          KeyType: HASH
      TimeToLiveSpecification:
        AttributeName: expires_at
        Enabled: true

  CostsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub '{runtime}costs-${{Environment}}'
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: date_key
          AttributeType: S
        - AttributeName: timestamp
          AttributeType: 'N'
      KeySchema:
        - AttributeName: date_key
          KeyType: HASH
        - AttributeName: timestamp
          KeyType: RANGE

Outputs:
  ApiEndpoint:
    Description: API Gateway endpoint URL
    Value: !Sub 'https://${{ServerlessRestApi}}.execute-api.${{AWS::Region}}.amazonaws.com/Prod/'
  FunctionArn:
    Description: Lambda function ARN
    Value: !GetAtt LambdaLLMFunction.Arn
"""
        return template

    def _build_cdk_stack(self) -> str:
        """Build CDK stack Python code."""
        project_name = self.config.project.name
        table_prefix = self.config.defaults.state.table_prefix

        return f'''"""CDK Stack for {project_name} - Generated by LambdaLLM.

Deploy with:
    cdk deploy --context env=dev
"""

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
)
from constructs import Construct


class LambdaLLMStack(Stack):
    """Infrastructure stack for {project_name}."""

    def __init__(self, scope: Construct, construct_id: str, env: str = "dev", **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # DynamoDB: Sessions table
        sessions_table = dynamodb.Table(
            self, "SessionsTable",
            table_name=f"{table_prefix}sessions-{{env}}",
            partition_key=dynamodb.Attribute(
                name="session_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            time_to_live_attribute="expires_at",
        )

        # DynamoDB: Costs table
        costs_table = dynamodb.Table(
            self, "CostsTable",
            table_name=f"{table_prefix}costs-{{env}}",
            partition_key=dynamodb.Attribute(
                name="date_key", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.NUMBER
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Lambda function
        handler = lambda_.Function(
            self, "LambdaLLMHandler",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handlers.main.lambda_handler",
            code=lambda_.Code.from_asset("."),
            timeout=Duration.seconds({self.config.defaults.timeout}),
            memory_size={self.config.defaults.memory},
            tracing=lambda_.Tracing.ACTIVE,
            environment={{
                "LAMBDALLM_ENV": env,
                "LAMBDALLM_TABLE_PREFIX": "{table_prefix}",
            }},
        )

        # Grant permissions
        sessions_table.grant_read_write_data(handler)
        costs_table.grant_read_write_data(handler)
        handler.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=["*"],
        ))

        # API Gateway
        api = apigw.RestApi(
            self, "LambdaLLMApi",
            rest_api_name=f"{project_name}-{{env}}",
            deploy_options=apigw.StageOptions(stage_name=env),
        )
        api.root.add_method("POST", apigw.LambdaIntegration(handler))
        api.root.add_proxy(default_integration=apigw.LambdaIntegration(handler))

        # Outputs
        CfnOutput(self, "ApiEndpoint", value=api.url)
        CfnOutput(self, "FunctionArn", value=handler.function_arn)
'''
