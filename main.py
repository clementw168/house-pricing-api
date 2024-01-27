from src.app import AppCreator
from src.services import dao, inference_service

app = AppCreator(dao, inference_service).get_app()
