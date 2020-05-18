"""
Setup for coloredlogs
"""
import logging
import coloredlogs


L = logging.getLogger("Courier")
fmt = "%(asctime)s:%(msecs)03d %(name)s [%(filename)s:%(lineno)s] %(levelname)s %(message)s"
coloredlogs.install(level="DEBUG", logger=L, fmt=fmt)
