"""
This module holds the Factory classes needed to handle the different data sources for each loop
"""
from src.parsers.base_parser import BaseCANParser
from src.parsers.openlka import OpenLKAParser
from src.parsers.recan import ReCANParser


class ParserFactory:
    @staticmethod
    def get_parser(source_dataset):
        if source_dataset == 'recan':
            return ReCANParser()
        elif source_dataset == 'openlka':
            return OpenLKAParser()
        else:
            return None
