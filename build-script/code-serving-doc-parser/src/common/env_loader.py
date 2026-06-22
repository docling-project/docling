import os
import glob

PROFILE = os.getenv("PROFILE", "local")  # prod / local

def get_env_file_list(module_dir: str, root_dir: str):
    """
    로드할 .env 파일 목록 정의 (리스트의 뒤에 있는 파일이 앞의 설정을 override)
    
    :param module_dir: 각 모듈의 최상위 디렉토리 (예: /Users/.../admin-api)
    :param root_dir: 전체 프로젝트의 최상위 루트 디렉토리 (예: /Users/.../GenOS)
    """
    files = []
    module_env_dir = os.path.join(module_dir, 'env')

    # PROFILE에 따라 로드할 프로파일 순서 결정
    if PROFILE == 'prod':
        profiles = ['prod']
    elif PROFILE == 'test':
        profiles = ['prod', 'dev', 'test']
    else:  # local
        profiles = ['prod', 'dev', 'local']

    # 1. 로컬 전용 프로젝트 공통 환경변수 파일 로드
    for profile in profiles:
        common_file = os.path.join(root_dir, f'.env.common.{profile}')
        if os.path.exists(common_file):
            files.append(common_file)

    # 2. 모듈 전용 환경변수 파일 로드
    for profile in profiles:
        module_file = os.path.join(module_env_dir, f'.env.{profile}')
        if os.path.exists(module_file):
            files.append(module_file)

    # 3. 로컬/테스트 전용 시크릿 환경변수 파일 로드 (.secrets/.env.*.secret.*)
    # - 클러스터(prod)에서는 사용하지 않고, 순전히 로컬 개발/테스트용 시크릿만 관리
    if PROFILE in ('local', 'test'):
        secrets_dir = os.path.join(root_dir, '.secrets')
        if os.path.isdir(secrets_dir):
            for profile in profiles:
                secret_pattern = os.path.join(secrets_dir, f'.env.*.secret.{profile}')
                for secret_file in sorted(glob.glob(secret_pattern)):
                    files.append(secret_file)

    return files

def customise_sources(
    init_settings,
    env_settings,
    dotenv_settings,
    file_secret_settings,
):
    """
    OS 환경변수와 .env 파일 우선순위 설정
    - PROFILE in ('local', 'test'): OS 환경변수를 사용하지 않고 .env / 파일 시크릿만 사용
      (로컬/테스트 실행 시 실수로 설정된 OS env 값이 테스트/로컬용 .env 값을 덮어쓰는 것을 방지)
    - 그 외(PROFILE=prod 등): OS 환경변수가 .env 파일보다 우선
      (쿠버네티스 ConfigMap이나 Deployment의 env 값이 로컬 파일을 덮어쓰기 위함)
    """
    if PROFILE in ('local', 'test'):
        return (init_settings, dotenv_settings, file_secret_settings)
    return (init_settings, env_settings, dotenv_settings, file_secret_settings)
