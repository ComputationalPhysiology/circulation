import pint

ureg = pint.UnitRegistry()


def mmHg_to_kPa(p):
    return p * 133.322 / 1000.0


def kPa_to_mmHg(p):
    return p * 1000.0 / 133.322
