from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    settings.create_directories()

    logger.info("%s starting", settings.app_name)
    logger.info("Environment: %s", settings.app_env)
    logger.info("Camera index: %s", settings.camera_index)
    logger.info("Project environment is ready")


if __name__ == "__main__":
    main()
