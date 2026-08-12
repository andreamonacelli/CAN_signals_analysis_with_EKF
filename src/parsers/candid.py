"""
This module will hold any data-parsing function specific to CANdid datasets (common data parsing functions are in the
generic /src/utils/datamanagement.py module)
"""


# The format handled by this parser is the following:
# (1718172626.613352) can0 141#2226242798822000
# (UNIX timestamp) channel CANid#payload
