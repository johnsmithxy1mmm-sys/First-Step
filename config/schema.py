"""Схема и валидация конфига объекта.

Pydantic-модели описывают структуру YAML, задают разумные значения по
умолчанию и проверяют осмысленность входа. Сообщения об ошибках — на русском,
по делу: что именно не так и что чинить. Без извинений и общих фраз.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Блоки конфига
# --------------------------------------------------------------------------- #
class ObjectInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    developer: str = ""
    zone: str = ""
    type: str = ""
    area_m2: float = Field(..., description="Площадь, м²")

    @field_validator("area_m2")
    @classmethod
    def _area_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("площадь объекта (object.area_m2) должна быть больше 0.")
        return v


class Tranche(BaseModel):
    model_config = ConfigDict(extra="forbid")

    month: int = Field(..., description="Месяц платежа от момента сделки")
    pct: float = Field(..., description="Доля от полной цены, %")

    @field_validator("month")
    @classmethod
    def _month_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("месяц транша (purchase.installment.schedule[].month) не может быть отрицательным.")
        return v

    @field_validator("pct")
    @classmethod
    def _pct_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("доля транша (purchase.installment.schedule[].pct) должна быть больше 0.")
        return v


class Installment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    down_payment_pct: Optional[float] = None
    schedule: List[Tranche]

    @model_validator(mode="after")
    def _check_schedule(self) -> "Installment":
        if not self.schedule:
            raise ValueError("график рассрочки (purchase.installment.schedule) пуст — добавьте хотя бы один транш.")
        total = sum(t.pct for t in self.schedule)
        # Сумма долей должна сходиться к 100 % (допуск на округление).
        if abs(total - 100.0) > 0.01:
            raise ValueError(
                f"сумма долей рассрочки = {total:.2f} %, должна быть 100 %. "
                f"Поправьте проценты в purchase.installment.schedule, чтобы они сходились к 100."
            )
        if self.down_payment_pct is not None:
            month0 = sum(t.pct for t in self.schedule if t.month == 0)
            if abs(month0 - self.down_payment_pct) > 0.01:
                raise ValueError(
                    f"первый взнос down_payment_pct = {self.down_payment_pct} % не совпадает с суммой "
                    f"траншей в месяце 0 ({month0} %). Приведите их в соответствие или уберите down_payment_pct."
                )
        return self


class Purchase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    price_rub: float
    installment: Installment

    @field_validator("price_rub")
    @classmethod
    def _price_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("цена объекта (purchase.price_rub) должна быть больше 0.")
        return v


class Operations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rental_rate_rub_month: float
    occupancy_pct: float
    opex_pct_of_revenue: float
    rental_growth_pct: float = 0.0
    # Месяц начала арендного дохода. По умолчанию 0 (доход со старта).
    # Для пресейла с рассрочкой «до ввода» поставьте месяц ввода в эксплуатацию.
    rent_start_month: int = 0

    @field_validator("rental_rate_rub_month")
    @classmethod
    def _rate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ставка аренды (operations.rental_rate_rub_month) должна быть больше 0.")
        return v

    @field_validator("occupancy_pct")
    @classmethod
    def _occupancy_range(cls, v: float) -> float:
        if not (0 < v <= 100):
            raise ValueError("загрузка (operations.occupancy_pct) должна быть в диапазоне (0; 100].")
        return v

    @field_validator("opex_pct_of_revenue")
    @classmethod
    def _opex_range(cls, v: float) -> float:
        if not (0 <= v < 100):
            raise ValueError("доля OPEX (operations.opex_pct_of_revenue) должна быть в диапазоне [0; 100).")
        return v

    @field_validator("rent_start_month")
    @classmethod
    def _rent_start_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("месяц начала аренды (operations.rent_start_month) не может быть отрицательным.")
        return v


class Exit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hold_years: int
    exit_cap_rate_pct: Optional[float] = None
    exit_price_growth_pct: Optional[float] = None
    selling_cost_pct: float = 0.0

    @field_validator("hold_years")
    @classmethod
    def _hold_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("горизонт удержания (exit.hold_years) должен быть больше 0.")
        return v

    @field_validator("selling_cost_pct")
    @classmethod
    def _selling_range(cls, v: float) -> float:
        if not (0 <= v < 100):
            raise ValueError("издержки продажи (exit.selling_cost_pct) должны быть в диапазоне [0; 100).")
        return v

    @model_validator(mode="after")
    def _check_exit_method(self) -> "Exit":
        has_cap = self.exit_cap_rate_pct is not None
        has_growth = self.exit_price_growth_pct is not None and self.exit_price_growth_pct > 0
        if not has_cap and not has_growth:
            raise ValueError(
                "не задан способ оценки выхода: укажите exit.exit_cap_rate_pct "
                "(капитализация дохода) либо exit.exit_price_growth_pct (рост цены)."
            )
        if has_cap and self.exit_cap_rate_pct <= 0:
            raise ValueError("exit.exit_cap_rate_pct должен быть больше 0.")
        return self


class Assumptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discount_rate_pct: float
    target_irr_pct: float

    @field_validator("discount_rate_pct")
    @classmethod
    def _discount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ставка дисконтирования (assumptions.discount_rate_pct) должна быть больше 0.")
        return v


class RangeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dist: Literal["triangular", "normal", "uniform"]
    low: Optional[float] = None
    mode: Optional[float] = None
    high: Optional[float] = None
    mean: Optional[float] = None
    sd: Optional[float] = None

    @model_validator(mode="after")
    def _check_params(self) -> "RangeSpec":
        if self.dist == "triangular":
            for f in ("low", "mode", "high"):
                if getattr(self, f) is None:
                    raise ValueError(f"для распределения triangular нужны поля low, mode, high (нет «{f}»).")
            if not (self.low <= self.mode <= self.high):
                raise ValueError("для triangular должно выполняться low ≤ mode ≤ high.")
        elif self.dist == "uniform":
            for f in ("low", "high"):
                if getattr(self, f) is None:
                    raise ValueError(f"для распределения uniform нужны поля low, high (нет «{f}»).")
            if self.low >= self.high:
                raise ValueError("для uniform должно выполняться low < high.")
        elif self.dist == "normal":
            for f in ("mean", "sd"):
                if getattr(self, f) is None:
                    raise ValueError(f"для распределения normal нужны поля mean, sd (нет «{f}»).")
            if self.sd <= 0:
                raise ValueError("для normal стандартное отклонение sd должно быть больше 0.")
        return self


class MonteCarlo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations: int = 10000
    seed: int = 42
    ranges: dict[str, RangeSpec] = Field(default_factory=dict)

    @field_validator("iterations")
    @classmethod
    def _iterations_positive(cls, v: int) -> int:
        if v < 100:
            raise ValueError("число итераций Монте-Карло (monte_carlo.iterations) должно быть не меньше 100.")
        return v


class Advisor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation: str
    thesis: str = ""
    exit_scenario: str = ""
    name: str = "Антон Перфилов"
    title: str = "Independent Real Estate Counsel"

    @field_validator("recommendation")
    @classmethod
    def _known_recommendation(cls, v: str) -> str:
        allowed = {"ACQUIRE", "HOLD", "AVOID"}
        if v not in allowed:
            raise ValueError(
                f"рекомендация (advisor.recommendation) = «{v}» не распознана. "
                f"Допустимы: ACQUIRE (Приобретать), HOLD (Удерживать), AVOID (Воздержаться)."
            )
        return v


class ObjectConfig(BaseModel):
    """Корневая модель конфига объекта."""

    model_config = ConfigDict(extra="forbid")

    object: ObjectInfo
    purchase: Purchase
    operations: Operations
    exit: Exit
    assumptions: Assumptions
    monte_carlo: MonteCarlo = Field(default_factory=MonteCarlo)
    advisor: Advisor
