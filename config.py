
# Pranav Minasandra
# pminasandra.github.io
# October 26, 2026

import os
import os.path
import sys

#Directories
try:
    PROJECTROOT = open('.cw', 'r').read().rstrip()
except FileNotFoundError:
    PROJECTROOT = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
DATA = os.path.join(PROJECTROOT, 'Data')
FIGURES = os.path.join(PROJECTROOT, 'Figures')

formats=['png', 'pdf', 'svg']

#Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False
