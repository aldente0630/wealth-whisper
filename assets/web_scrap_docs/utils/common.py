def get_name(proj_name: str, stage: str, resource_value: str) -> str:
    return "-".join([proj_name, stage, resource_value])


def get_ssm_param_key(
    proj_name: str, stage: str, resource_name: str, resource_value: str
) -> str:
    return f"/{proj_name}/{stage}/{resource_name}/{resource_value}"
