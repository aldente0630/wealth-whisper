import json
import os
import platform
import sys
from typing import Optional
import aws_cdk as core
from aws_cdk import (
    aws_batch as batch,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ecr_assets as ecr_assets,
    aws_elasticloadbalancingv2 as elbv2,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_opensearchservice as opensearch,
    aws_s3 as s3,
    aws_sagemaker as sagemaker,
    aws_secretsmanager as secretsmanager,
    aws_ssm as ssm,
)
from constructs import Construct
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.utils import (
    get_model_id,
    get_name,
    get_provider_name,
    get_ssm_param_key,
    load_config,
    parse_cron_expr,
)


class RagBaseStack(core.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        proj_name: Optional[str] = None,
        stage: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # Create a default bucket
        bucket_name = (
            f"{get_name(proj_name, stage, 'default')}-{str(core.Aws.ACCOUNT_ID)}"
        )
        removal_policy = (
            core.RemovalPolicy.DESTROY
            if stage.lower() == "dev"
            else core.RemovalPolicy.RETAIN
        )

        default_bucket = s3.Bucket(
            self,
            "DefaultBucket",
            bucket_name=bucket_name,
            removal_policy=removal_policy,
        )

        ssm.StringParameter(
            self,
            "DefaultBucketName",
            parameter_name=get_ssm_param_key(proj_name, stage, "s3_bucket", "default"),
            string_value=default_bucket.bucket_name,
        )

        # Create a default VPC and subnets and a default security group
        vpc_name = get_name(proj_name, stage, "default")
        subnet_configuration = [
            ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC),
            ec2.SubnetConfiguration(
                name="private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            ec2.SubnetConfiguration(
                name="isolated", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED
            ),
        ]

        default_vpc = ec2.Vpc(
            self,
            "DefaultVpc",
            ip_addresses=ec2.IpAddresses.cidr("10.0.0.0/16"),
            subnet_configuration=subnet_configuration,
            vpc_name=vpc_name,
        )

        core.CfnOutput(
            self,
            "DefaultVpcId",
            value=default_vpc.vpc_id,
            export_name=f"{proj_name}-{stage}-default-vpc-id",
        )
        core.CfnOutput(
            self,
            "DefaultAvailabilityZones",
            value=core.Fn.join(",", default_vpc.availability_zones),
            export_name=f"{proj_name}-{stage}-default-availability-zones",
        )
        core.CfnOutput(
            self,
            "DefaultPublicSubnetIds",
            value=core.Fn.join(
                ",",
                list(map(lambda x: x.subnet_id, default_vpc.public_subnets)),
            ),
            export_name=f"{proj_name}-{stage}-default-public-subnet-ids",
        )
        core.CfnOutput(
            self,
            "DefaultPrivateSubnetIds",
            value=core.Fn.join(
                ",",
                list(map(lambda x: x.subnet_id, default_vpc.private_subnets)),
            ),
            export_name=f"{proj_name}-{stage}-default-private-subnet-ids",
        )

        security_group_name = get_name(proj_name, stage, "default")
        default_security_group = ec2.SecurityGroup(
            self,
            "DefaultSecurityGroup",
            vpc=default_vpc,
            allow_all_outbound=True,
            security_group_name=security_group_name,
        )
        core.CfnOutput(
            self,
            "DefaultSecurityGroupId",
            value=default_security_group.security_group_id,
            export_name=f"{proj_name}-{stage}-default-security-group-id",
        )


class RagFrontendStack(core.Stack):
    def __init__(
        self,
        scope: Optional[Construct] = None,
        id: Optional[str] = None,
        proj_name: Optional[str] = None,
        stage: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        vpc = ec2.Vpc.from_vpc_attributes(
            self,
            "DefaultVpc",
            vpc_id=core.Fn.import_value(f"{proj_name}-{stage}-default-vpc-id"),
            availability_zones=core.Fn.split(
                ",",
                core.Fn.import_value(f"{proj_name}-{stage}-default-availability-zones"),
            ),
            public_subnet_ids=core.Fn.split(
                ",",
                core.Fn.import_value(f"{proj_name}-{stage}-default-public-subnet-ids"),
            ),
            private_subnet_ids=core.Fn.split(
                ",",
                core.Fn.import_value(f"{proj_name}-{stage}-default-private-subnet-ids"),
            ),
        )

        # Create a repository for the frontend
        asset_name = get_name(proj_name, stage, "frontend")
        docker_image_asset = ecr_assets.DockerImageAsset(
            self,
            "FrontendImageAsset",
            asset_name=asset_name,
            directory=os.path.abspath(get_dir_path(os.path.join(os.pardir, "app"))),
        )

        # Creating a role for the frontend
        role_name = get_name(proj_name, stage, "frontend")
        managed_policies = [
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBedrockFullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonEC2ContainerRegistryFullAccess"
            ),
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonOpenSearchServiceFullAccess"
            ),
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMFullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name("SecretsManagerReadWrite"),
        ]

        frontend_role = iam.Role(
            self,
            "FrontendRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=managed_policies,
            role_name=role_name,
        )

        # Create a service for the frontend
        platform_map = {
            "x86_64": ecs.CpuArchitecture.X86_64,
            "arm64": ecs.CpuArchitecture.ARM64,
        }
        architecture = platform_map[platform.machine()]

        cluster_name = get_name(proj_name, stage, "frontend")
        frontend_cluster = ecs.Cluster(
            self, "FrontEndCluster", cluster_name=cluster_name, vpc=vpc
        )

        load_balancer_name = get_name(proj_name, stage, "frontend")
        service_name = load_balancer_name
        frontend_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "FrontendService",
            task_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            cluster=frontend_cluster,
            load_balancer_name=load_balancer_name,
            public_load_balancer=True,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=architecture,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
            service_name=service_name,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                container_port=8501,
                image=ecs.ContainerImage.from_docker_image_asset(docker_image_asset),
                task_role=frontend_role,
            ),
        )
        frontend_service.target_group.configure_health_check(path="/healthz")

        # Create a distribution for the frontend
        custom_header_key = "X-Verify-Origin"
        custom_header_value = "-".join((self.stack_name, "FrontendDistribution"))

        origin = origins.LoadBalancerV2Origin(
            frontend_service.load_balancer,
            http_port=80,
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
            origin_path="/",
            custom_headers={custom_header_key: custom_header_value},
        )

        frontend_distribution = cloudfront.Distribution(
            self,
            "FrontendDistribution",
            default_behavior=cloudfront.BehaviorOptions(
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                compress=False,
                origin=origin,
                origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_AND_CLOUDFRONT_2022,
                response_headers_policy=cloudfront.ResponseHeadersPolicy.CORS_ALLOW_ALL_ORIGINS,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            ),
            minimum_protocol_version=cloudfront.SecurityPolicyProtocol.SSL_V3,
        )

        core.CfnOutput(
            self,
            "FrontendDomainName",
            value=f"https://{frontend_distribution.domain_name}",
            export_name=f"{proj_name}-{stage}-frontend-domain-name",
        )

        # Create listener rules for the frontend
        elbv2.ApplicationListenerRule(
            self,
            "FrontendHeaderListenerRule",
            listener=frontend_service.listener,
            priority=1,
            action=elbv2.ListenerAction.forward([frontend_service.target_group]),
            conditions=[
                elbv2.ListenerCondition.http_header(
                    custom_header_key, [custom_header_value]
                )
            ],
        )
        elbv2.ApplicationListenerRule(
            self,
            "FrontendRedirectListenerRule",
            listener=frontend_service.listener,
            priority=5,
            action=elbv2.ListenerAction.redirect(
                host=frontend_distribution.domain_name,
                permanent=True,
                port="443",
                protocol="HTTPS",
            ),
            conditions=[elbv2.ListenerCondition.path_patterns(["*"])],
        )


class RagIndexingStack(core.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        proj_name: Optional[str] = None,
        stage: Optional[str] = None,
        cron_expr: Optional[str] = None,
        host_pdf_reader: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # Create a role for web scraping function
        role_name = get_name(proj_name, stage, "scraping")
        managed_policies = [
            iam.ManagedPolicy.from_aws_managed_policy_name("AWSBatchFullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonDynamoDBFullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2FullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMFullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            ),
        ]

        scrap_role = iam.Role(
            self,
            "ScrapingRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=managed_policies,
            role_name=role_name,
        )

        # Create a function for web scraping from local code
        function_name = get_name(proj_name, stage, "scraping")
        code = lambda_.DockerImageCode.from_image_asset(
            get_dir_path(os.path.join(os.pardir, "assets", "web_scrap_docs")),
            build_args={"--platform": "linux/amd64"},
        )
        environment = {
            "PROJ_NAME": proj_name,
            "REGION_NAME": str(core.Aws.REGION),
            "STAGE": stage,
        }
        vpc = ec2.Vpc.from_vpc_attributes(
            self,
            "DefaultVpc",
            vpc_id=core.Fn.import_value(f"{proj_name}-{stage}-default-vpc-id"),
            availability_zones=core.Fn.split(
                ",",
                core.Fn.import_value(f"{proj_name}-{stage}-default-availability-zones"),
            ),
            private_subnet_ids=core.Fn.split(
                ",",
                core.Fn.import_value(f"{proj_name}-{stage}-default-private-subnet-ids"),
            ),
        )

        scrap_function = lambda_.DockerImageFunction(
            self,
            "ScrapingFunction",
            code=code,
            environment=environment,
            ephemeral_storage_size=core.Size.mebibytes(1024),
            function_name=function_name,
            memory_size=256,
            role=scrap_role,
            timeout=core.Duration.seconds(900),
            vpc=vpc,
            retry_attempts=2,
        )

        # Create a triggering rule for web scraping (starts at a specific time)
        rule_name = get_name(proj_name, stage, "scraping")

        _ = events.Rule(
            self,
            "ScrapingRule",
            schedule=events.Schedule.cron(**parse_cron_expr(cron_expr)),
            targets=[targets.LambdaFunction(scrap_function)],
            rule_name=rule_name,
        )

        # Create a job queue for data ingestion
        job_queue_name = get_name(proj_name, stage, "ingestion")
        ingestion_job_queue = batch.JobQueue(
            self,
            "IngestionJobQueue",
            job_queue_name=job_queue_name,
        )

        # Create a role for data ingestion container
        role_name = get_name(proj_name, stage, "ingestion")
        managed_policies = [
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBedrockFullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonDynamoDBFullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonOpenSearchServiceFullAccess"
            ),
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMFullAccess"),
            iam.ManagedPolicy.from_aws_managed_policy_name("SecretsManagerReadWrite"),
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AmazonEC2ContainerServiceforEC2Role"
            ),
        ]

        ingestion_role = iam.Role(
            self,
            "IngestionRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=managed_policies,
            role_name=role_name,
        )

        # Create a compute environment for data ingestion
        compute_environment_name = get_name(proj_name, stage, "ingestion")
        compute_environment = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "IngestionComputeEnvironment",
            instance_role=ingestion_role,
            instance_types=[
                ec2.InstanceType.of(ec2.InstanceClass.M6I, ec2.InstanceSize.XLARGE2),
                ec2.InstanceType.of(ec2.InstanceClass.M6I, ec2.InstanceSize.XLARGE4),
                ec2.InstanceType.of(ec2.InstanceClass.M6I, ec2.InstanceSize.XLARGE8),
                ec2.InstanceType.of(ec2.InstanceClass.M6I, ec2.InstanceSize.XLARGE16),
            ],
            use_optimal_instance_classes=False,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            compute_environment_name=compute_environment_name,
        )
        ingestion_job_queue.add_compute_environment(compute_environment, 1)

        # Create a job definition for data ingestion
        job_definition_name = get_name(proj_name, stage, "ingestion")
        ingestion_job_definition = batch.EcsJobDefinition(
            self,
            "IngestionJobDefinition",
            container=batch.EcsEc2ContainerDefinition(
                self,
                "IngestionContainerDefinition",
                cpu=2,
                image=ecs.ContainerImage.from_asset(
                    get_dir_path(os.path.join(os.pardir, "assets", "ingest_data")),
                    build_args={"--platform": "linux/amd64"},
                ),
                memory=core.Size.mebibytes(2048),
                command=[
                    "python",
                    "app.py",
                    "--start-date",
                    "Ref::start_date",
                    "--end-date",
                    "Ref::end_date",
                ],
                environment={
                    "PROJ_NAME": proj_name,
                    "REGION_NAME": str(core.Aws.REGION),
                    "STAGE": stage,
                },
            ),
            job_definition_name=job_definition_name,
        )

        ssm.StringParameter(
            self,
            "IngestionJobQueueName",
            parameter_name=get_ssm_param_key(
                proj_name, stage, "batch_job_queue", "ingestion"
            ),
            string_value=ingestion_job_queue.job_queue_name,
        )
        ssm.StringParameter(
            self,
            "IngestionJobDefinitionName",
            parameter_name=get_ssm_param_key(
                proj_name, stage, "batch_job_definition", "ingestion"
            ),
            string_value=ingestion_job_definition.job_definition_name,
        )

        if host_pdf_reader:
            # Create a service for documents parsing
            cluster_name = get_name(proj_name, stage, "parsing")
            cluster = ecs.Cluster(
                self, "ParsingCluster", cluster_name=cluster_name, vpc=vpc
            )

            load_balancer_name = get_name(proj_name, stage, "parsing")
            service_name = load_balancer_name
            parsing_service = ecs_patterns.ApplicationLoadBalancedFargateService(
                self,
                "ParsingService",
                task_subnets=ec2.SubnetSelection(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
                ),
                cluster=cluster,
                listener_port=5001,
                load_balancer_name=load_balancer_name,
                protocol=elbv2.ApplicationProtocol.HTTP,
                public_load_balancer=False,
                service_name=service_name,
                task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                    container_port=5001,
                    image=ecs.ContainerImage.from_registry(
                        "ghcr.io/nlmatics/nlm-ingestor:latest"
                    ),
                ),
                cpu=4096,
                memory_limit_mib=8192,
            )
            parsing_service.target_group.configure_health_check(
                healthy_http_codes="404"
                # Due to non-implementation of health check in the docker image
            )

            ssm.StringParameter(
                self,
                "ParsingAlbDnsName",
                parameter_name=get_ssm_param_key(
                    proj_name, stage, "alb_dns", "parsing"
                ),
                string_value=parsing_service.load_balancer.load_balancer_dns_name,
            )


class RagSearchStack(core.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        proj_name: Optional[str] = None,
        stage: Optional[str] = None,
        deploy_model: bool = False,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # Create a domain for documents
        domain_name = get_name(proj_name, stage, "docs")
        docs_domain_secret = secretsmanager.Secret(
            self,
            "DocsDomainSecret",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_characters='/@"',
                generate_string_key="opensearch_password",
                secret_string_template=json.dumps(
                    {"opensearch_username": os.getenv("opensearch_username", "user")}
                ),
            ),
            secret_name=get_name(proj_name, stage, "docs"),
        )

        access_policies = [
            iam.PolicyStatement(
                actions=["es:*"],
                principals=[iam.ArnPrincipal("*")],
                resources=["*"],
            )
        ]
        capacity = (
            opensearch.CapacityConfig(
                data_node_instance_type="r6g.large.search",
                data_nodes=1,
            )
            if stage.lower() == "dev"
            else opensearch.CapacityConfig(
                data_node_instance_type="r6g.large.search",
                data_nodes=3,
                master_node_instance_type="r6g.large.search",
                master_nodes=3,
                multi_az_with_standby_enabled=True,
            )
        )
        ebs = opensearch.EbsOptions(
            volume_size=10, volume_type=ec2.EbsDeviceVolumeType.GP3
        )
        fine_grained_access_control = opensearch.AdvancedSecurityOptions(
            master_user_name=docs_domain_secret.secret_value_from_json(
                "opensearch_username"
            ).unsafe_unwrap(),
            master_user_password=docs_domain_secret.secret_value_from_json(
                "opensearch_password"
            ),
        )
        removal_policy = (
            core.RemovalPolicy.DESTROY
            if stage.lower() == "dev"
            else core.RemovalPolicy.RETAIN
        )
        vpc = ec2.Vpc.from_vpc_attributes(
            self,
            "DefaultVpc",
            vpc_id=core.Fn.import_value(f"{proj_name}-{stage}-default-vpc-id"),
            availability_zones=core.Fn.split(
                ",",
                core.Fn.import_value(f"{proj_name}-{stage}-default-availability-zones"),
            ),
            private_subnet_ids=core.Fn.split(
                ",",
                core.Fn.import_value(f"{proj_name}-{stage}-default-private-subnet-ids"),
            ),
        )
        subnets = vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
        ).subnets
        subnets = [subnets[0]] if stage.lower() == "dev" else [subnets]

        _ = opensearch.Domain(
            self,
            "DocsDomain",
            version=opensearch.EngineVersion.OPENSEARCH_2_11,
            access_policies=access_policies,
            capacity=capacity,
            domain_name=domain_name,
            ebs=ebs,
            fine_grained_access_control=fine_grained_access_control,
            encryption_at_rest=opensearch.EncryptionAtRestOptions(enabled=True),
            enforce_https=True,
            node_to_node_encryption=True,
            removal_policy=removal_policy,
            vpc_subnets=[ec2.SubnetSelection(subnets=subnets)],
        )

        # Create a table for metadata and an index on it
        index_name = get_name(proj_name, stage, "metadata")
        global_secondary_indexes = [
            dynamodb.GlobalSecondaryIndexPropsV2(
                index_name=index_name,
                partition_key=dynamodb.Attribute(
                    name="Category",
                    type=dynamodb.AttributeType.STRING,
                ),
                sort_key=dynamodb.Attribute(
                    name="BaseDate", type=dynamodb.AttributeType.STRING
                ),
            )
        ]

        table_name = get_name(proj_name, stage, "metadata")

        metadata_table = dynamodb.TableV2(
            self,
            "MetadataTable",
            partition_key=dynamodb.Attribute(
                name="DocId", type=dynamodb.AttributeType.STRING
            ),
            global_secondary_indexes=global_secondary_indexes,
            removal_policy=removal_policy,
            table_name=table_name,
        )

        ssm.StringParameter(
            self,
            "MetadataTableName",
            parameter_name=get_ssm_param_key(proj_name, stage, "ddb_table", "metadata"),
            string_value=metadata_table.table_name,
        )
        ssm.StringParameter(
            self,
            "MetadataIndexName",
            parameter_name=get_ssm_param_key(proj_name, stage, "ddb_index", "metadata"),
            string_value=global_secondary_indexes[0].index_name,
        )

        if deploy_model:
            # Create an endpoint for the embedding model
            role_name = get_name(proj_name, stage, "deployment")
            managed_policies = [
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
            ]

            deployment_role = iam.Role(
                self,
                "DeploymentRole",
                assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
                managed_policies=managed_policies,
                role_name=role_name,
            )

            primary_container = {
                "image": get_hf_infer_container_url(str(core.Aws.REGION), "2.1.0"),
                "environment": {
                    "HF_MODEL_ID": get_model_id(model_name),
                    "HF_TASK": "feature-extraction",
                },
            }
            vpc_config = sagemaker.CfnModel.VpcConfigProperty(
                security_group_ids=[
                    core.Fn.import_value(
                        f"{proj_name}-{stage}-default-security-group-id"
                    )
                ],
                subnets=core.Fn.split(
                    ",",
                    core.Fn.import_value(
                        f"{proj_name}-{stage}-default-private-subnet-ids"
                    ),
                ),
            )

            embedding_model = sagemaker.CfnModel(
                self,
                "EmbeddingModel",
                execution_role_arn=deployment_role.role_arn,
                primary_container=primary_container,
                vpc_config=vpc_config,
            )

            endpoint_config_name = get_name(proj_name, stage, "embedding")
            production_variants = [
                {
                    "modelName": embedding_model.attr_model_name,
                    "initialInstanceCount": 1,
                    "instanceType": "ml.m5.xlarge",
                    "variantName": "AllTraffic",
                }
            ]

            embedding_endpoint_config = sagemaker.CfnEndpointConfig(
                self,
                "EmbeddingEndpointConfig",
                production_variants=production_variants,
                endpoint_config_name=endpoint_config_name,
            )

            endpoint_name = get_name(proj_name, stage, "embedding")

            _ = sagemaker.CfnEndpoint(
                self,
                "EmbeddingEndpoint",
                endpoint_config_name=embedding_endpoint_config.attr_endpoint_config_name,
                endpoint_name=endpoint_name,
            )


def get_dir_path(dir_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))


def get_hf_infer_container_url(
    region_name: str,
    pytorch_version: str,
    use_gpu: bool = False,
) -> str:
    hf_infer_container_url_dict = (
        {
            "2.1.0": f"763104351884.dkr.ecr.{region_name}.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04",
        }
        if use_gpu
        else {
            "2.1.0": f"763104351884.dkr.ecr.{region_name}.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04",
        }
    )

    return hf_infer_container_url_dict[pytorch_version]


if __name__ == "__main__":
    config_dir = get_dir_path(os.path.join(os.pardir, "app", "configs"))
    config = load_config(os.path.join(config_dir, "config.yaml"))
    load_dotenv()

    deploy_model = get_provider_name(get_model_id(config.embedding_model_name)) not in (
        "amazon",
        "cohere",
    )

    env = core.Environment(account=core.Aws.ACCOUNT_ID, region=config.region_name)
    app = core.App()

    RagBaseStack(
        app, "RagBaseStack", proj_name=config.proj_name, stage=config.stage, env=env
    )

    RagSearchStack(
        app,
        "RagSearchStack",
        proj_name=config.proj_name,
        stage=config.stage,
        deploy_model=deploy_model,
        model_name=config.embedding_model_name,
        env=env,
    )

    RagIndexingStack(
        app,
        "RagIndexingStack",
        proj_name=config.proj_name,
        stage=config.stage,
        cron_expr=config.cron_expr,
        host_pdf_reader=(config.pdf_parser_type == "layout_pdf_reader"),
        env=env,
    )

    RagFrontendStack(
        app,
        "RagFrontendStack",
        proj_name=config.proj_name,
        stage=config.stage,
        env=env,
    )

    app.synth()
