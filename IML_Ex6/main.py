"""
Michael Zhitomirsky 321962714
Or Shahar           208712471
"""

from Parity3 import Xor3

if __name__ == '__main__':
    Xor3_section_a = Xor3()
    Xor3_section_a.run_iteration()
    Xor3_section_a.plot()

    Xor3_section_b = Xor3(6)
    Xor3_section_b.run_iteration()
    Xor3_section_b.plot()
