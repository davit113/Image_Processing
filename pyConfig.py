
MODULES = "code"
DIRECTORY_SEPERATOR = "//"

from sys import path
from os import getcwd

CURRENT_PATH = getcwd()
path.insert(1, CURRENT_PATH + DIRECTORY_SEPERATOR + MODULES)

FONT_MAIN = 'Comic Sans'
FONT_COLOR_MAIN = '#5F1'
BTN_COLOR_MAIN = '#fcc'
BG_COLOR_MAIN = 'BLACK'
BUTTON_NAMES = ['accure image', 'find objects', 'show target size', 'calculate distance']
