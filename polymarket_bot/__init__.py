"""Polymarket longshot bot — барбелл на мисспрайсинге хвостовых исходов.

Сканер мисспрайсинга: покупает лонгшот только когда собственная оценка
вероятности существенно выше рыночной цены (p_est / p_mkt >= порога).
Модули: scanner -> estimator (base rates, когерентность, LLM, momentum)
-> portfolio (fractional Kelly + лимиты) -> executor (maker-лимитки)
-> ledger (sqlite, атрибуция PnL) -> monitor (rich + Telegram).
"""

__version__ = "1.0.0"
