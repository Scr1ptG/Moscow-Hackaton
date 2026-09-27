"""Конфигурация Sphinx: документация по коду ML-ядра, NDTP и ML-сервиса."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Transit Delay Predictor"
author = "Команда хакатона"
language = "ru"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode", "sphinx.ext.napoleon"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = []
html_theme = "alabaster"
exclude_patterns = ["_build", "html", "openapi"]
