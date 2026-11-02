"""Моторная кора REGENESIS (Законы 1–2 манифеста) + Двухпутевой слогово-фонемный декодер v8.0."""
from .pool import MotorPool, MotorOut
from .syllabary_decoder import BGPQSyllabaryDecoder, SyllabicDecodeOut

__all__ = ["MotorPool", "MotorOut", "BGPQSyllabaryDecoder", "SyllabicDecodeOut"]
