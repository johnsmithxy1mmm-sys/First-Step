#!/usr/bin/env python3
"""Графический интерфейс для сборки инвестиционных меморандумов.

Окно с формой параметров объекта. Заполняете поля → «Сформировать PDF» →
готовый меморандум в папке out/. Можно загрузить/сохранить конфиг как YAML.

Запуск из исходников:  python app.py
В собранном виде:       двойной клик по Меморандумы.exe
"""

from __future__ import annotations

import datetime as _dt
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from appcore import (REC_OPTIONS, app_dir, config_to_flat, config_to_yaml,
                     generate, open_file, slug, yaml_to_dict)
from config.loader import ConfigError

# ── Бренд-палитра окна (зеркало отчёта) ─────────────────────────────────── #
PAPER = "#F5F1E8"
CARD = "#FBF8F1"
INK = "#211C18"
INK_SOFT = "#5C544A"
BRASS = "#9A7B4F"
BRASS_DEEP = "#7E6438"
LINE = "#DAD2C2"

# Значения по умолчанию — параметры из objects/park.yaml.
DEFAULTS = {
    "name": "МФК «Парк»", "developer": "ГК «Десо»", "zone": "Сочи · Центральный",
    "type": "апарт-комплекс", "area_m2": "34",
    "price_rub": "18900000", "down_payment_pct": "30",
    "schedule": [("0", "30"), ("12", "35"), ("24", "35")],
    "rental_rate_rub_month": "95000", "occupancy_pct": "70", "opex_pct_of_revenue": "28",
    "rental_growth_pct": "6", "rent_start_month": "24",
    "hold_years": "7", "exit_method": "growth", "exit_value": "8", "selling_cost_pct": "4",
    "discount_rate_pct": "18", "target_irr_pct": "15",
    "iterations": "10000", "seed": "42",
    "occ_low": "55", "occ_mode": "70", "occ_high": "82",
    "grow_mean": "6", "grow_sd": "2",
    "exitr_low": "5", "exitr_mode": "8", "exitr_high": "11",
    "recommendation": "HOLD — Удерживать",
    "thesis": "Рассрочка до ввода без удорожания; на текущем прайсе — удержание.",
    "exit_scenario": "Перепродажа на вводе, +18–22 %",
    "advisor_name": "Антон Перфилов", "advisor_title": "Independent Real Estate Counsel",
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Инвестиционные меморандумы — Антон Перфилов")
        self.geometry("780x900")
        self.configure(bg=PAPER)
        self.vars: dict[str, tk.StringVar] = {}
        self.schedule_rows: list[tuple[tk.StringVar, tk.StringVar, ttk.Frame]] = []
        self._busy = False

        self._setup_style()
        self._build_layout()
        self._set_form(DEFAULTS)

    # ── Оформление ─────────────────────────────────────────────────────── #
    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=PAPER)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=PAPER, foreground=INK, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=CARD, foreground=INK, font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background=CARD, foreground=INK_SOFT, font=("Segoe UI", 8))
        style.configure("Head.TLabel", background=PAPER, foreground=BRASS_DEEP,
                        font=("Georgia", 13, "bold"))
        style.configure("Group.TLabelframe", background=CARD, bordercolor=LINE, relief="solid")
        style.configure("Group.TLabelframe.Label", background=PAPER, foreground=BRASS_DEEP,
                        font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Go.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("TEntry", fieldbackground="white")

    # ── Скелет окна: прокручиваемая форма + нижняя панель ──────────────── #
    def _build_layout(self):
        header = ttk.Label(self, text="Формирование меморандума", style="Head.TLabel")
        header.pack(anchor="w", padx=16, pady=(12, 4))

        # Прокручиваемая область.
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=12)
        canvas = tk.Canvas(container, bg=PAPER, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.form = ttk.Frame(canvas)
        self.form.bind("<Configure>",
                       lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.form, anchor="nw", width=740)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        self._build_sections()
        self._build_bottom_bar()

    def _section(self, title: str) -> ttk.Frame:
        lf = ttk.Labelframe(self.form, text="  " + title + "  ", style="Group.TLabelframe")
        lf.pack(fill="x", pady=7, padx=2)
        inner = ttk.Frame(lf, style="Card.TFrame")
        inner.pack(fill="x", padx=10, pady=8)
        return inner

    def _row(self, parent, label: str, key: str, hint: str = "", width: int = 22):
        r = parent.grid_size()[1]
        ttk.Label(parent, text=label, style="Card.TLabel").grid(
            row=r, column=0, sticky="w", pady=3, padx=(0, 8))
        var = tk.StringVar()
        self.vars[key] = var
        ttk.Entry(parent, textvariable=var, width=width).grid(row=r, column=1, sticky="w", pady=3)
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel").grid(
                row=r, column=2, sticky="w", padx=8)
        parent.columnconfigure(2, weight=1)

    def _build_sections(self):
        # Объект.
        s = self._section("Объект")
        self._row(s, "Название", "name", width=34)
        self._row(s, "Застройщик", "developer", width=34)
        self._row(s, "Микрозона", "zone", width=34)
        self._row(s, "Тип", "type", width=34)
        self._row(s, "Площадь, м²", "area_m2")

        # Покупка и рассрочка.
        s = self._section("Покупка и рассрочка")
        self._row(s, "Цена объекта, ₽", "price_rub")
        self._row(s, "Первый взнос, %", "down_payment_pct", "= сумма траншей месяца 0")
        ttk.Label(s, text="График рассрочки (месяц · доля %):", style="Card.TLabel").grid(
            row=s.grid_size()[1], column=0, columnspan=3, sticky="w", pady=(8, 2))
        self.sched_frame = ttk.Frame(s, style="Card.TFrame")
        self.sched_frame.grid(row=s.grid_size()[1], column=0, columnspan=3, sticky="w")
        btns = ttk.Frame(s, style="Card.TFrame")
        btns.grid(row=s.grid_size()[1], column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Button(btns, text="+ транш", command=self._add_schedule_row).pack(side="left")
        self.sched_sum = ttk.Label(btns, text="", style="Hint.TLabel")
        self.sched_sum.pack(side="left", padx=12)

        # Операции.
        s = self._section("Операции")
        self._row(s, "Ставка аренды, ₽/мес", "rental_rate_rub_month")
        self._row(s, "Загрузка, %", "occupancy_pct")
        self._row(s, "Доля OPEX, %", "opex_pct_of_revenue", "УК, налоги, обслуживание")
        self._row(s, "Рост ставки, %/год", "rental_growth_pct")
        self._row(s, "Начало аренды, мес.", "rent_start_month", "0 = со старта; для пресейла — месяц ввода")

        # Выход.
        s = self._section("Выход")
        self._row(s, "Горизонт удержания, лет", "hold_years")
        ttk.Label(s, text="Метод оценки выхода:", style="Card.TLabel").grid(
            row=s.grid_size()[1], column=0, sticky="w", pady=(6, 2))
        self.vars["exit_method"] = tk.StringVar(value="growth")
        mrow = ttk.Frame(s, style="Card.TFrame")
        mrow.grid(row=s.grid_size()[1] - 1, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(mrow, text="Прирост цены, %/год", value="growth",
                        variable=self.vars["exit_method"], command=self._sync_exit_label).pack(side="left")
        ttk.Radiobutton(mrow, text="Cap rate, %", value="cap",
                        variable=self.vars["exit_method"], command=self._sync_exit_label).pack(side="left", padx=10)
        self._row(s, "Значение выхода", "exit_value")
        self._row(s, "Издержки продажи, %", "selling_cost_pct")

        # Допущения.
        s = self._section("Допущения")
        self._row(s, "Ставка дисконтирования, %", "discount_rate_pct")
        self._row(s, "Целевой IRR, %", "target_irr_pct")

        # Монте-Карло.
        s = self._section("Монте-Карло")
        self._row(s, "Итераций", "iterations")
        self._row(s, "Сид (для воспроизводимости)", "seed")
        self._triple(s, "Загрузка (треуг.): мин · мода · макс", "occ_low", "occ_mode", "occ_high")
        self._pair(s, "Рост ставки (нормал.): среднее · σ", "grow_mean", "grow_sd")
        self.exit_range_lbl = self._triple(
            s, "Выход (треуг.): мин · мода · макс", "exitr_low", "exitr_mode", "exitr_high")

        # Советник.
        s = self._section("Советник")
        ttk.Label(s, text="Рекомендация", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        self.vars["recommendation"] = tk.StringVar()
        ttk.Combobox(s, textvariable=self.vars["recommendation"], values=REC_OPTIONS,
                     state="readonly", width=30).grid(row=0, column=1, columnspan=2, sticky="w")
        self._row(s, "Тезис", "thesis", width=50)
        self._row(s, "Сценарий выхода", "exit_scenario", width=50)
        self._row(s, "Подпись — имя", "advisor_name", width=34)
        self._row(s, "Подпись — должность", "advisor_title", width=34)

    def _triple(self, parent, label, k1, k2, k3):
        r = parent.grid_size()[1]
        lbl = ttk.Label(parent, text=label, style="Card.TLabel")
        lbl.grid(row=r, column=0, sticky="w", pady=3)
        box = ttk.Frame(parent, style="Card.TFrame")
        box.grid(row=r, column=1, columnspan=2, sticky="w")
        for k in (k1, k2, k3):
            self.vars[k] = tk.StringVar()
            ttk.Entry(box, textvariable=self.vars[k], width=8).pack(side="left", padx=2)
        return lbl

    def _pair(self, parent, label, k1, k2):
        r = parent.grid_size()[1]
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=3)
        box = ttk.Frame(parent, style="Card.TFrame")
        box.grid(row=r, column=1, columnspan=2, sticky="w")
        for k in (k1, k2):
            self.vars[k] = tk.StringVar()
            ttk.Entry(box, textvariable=self.vars[k], width=8).pack(side="left", padx=2)

    def _sync_exit_label(self):
        method = self.vars["exit_method"].get()
        self.exit_range_lbl.configure(
            text=("Прирост цены (треуг.): мин · мода · макс" if method == "growth"
                  else "Cap rate (треуг.): мин · мода · макс"))

    # ── Динамические транши рассрочки ─────────────────────────────────── #
    def _add_schedule_row(self, month: str = "", pct: str = ""):
        row = ttk.Frame(self.sched_frame, style="Card.TFrame")
        row.pack(anchor="w", pady=1)
        mv, pv = tk.StringVar(value=month), tk.StringVar(value=pct)
        ttk.Label(row, text="мес.", style="Card.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=mv, width=6).pack(side="left", padx=(2, 8))
        ttk.Label(row, text="доля %", style="Card.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=pv, width=6).pack(side="left", padx=2)
        ttk.Button(row, text="✕", width=2,
                   command=lambda: self._del_schedule_row(row)).pack(side="left", padx=6)
        for v in (mv, pv):
            v.trace_add("write", lambda *_: self._update_sched_sum())
        self.schedule_rows.append((mv, pv, row))
        self._update_sched_sum()

    def _del_schedule_row(self, row):
        self.schedule_rows = [r for r in self.schedule_rows if r[2] is not row]
        row.destroy()
        self._update_sched_sum()

    def _update_sched_sum(self):
        total = 0.0
        for mv, pv, _ in self.schedule_rows:
            try:
                total += float(pv.get().replace(",", "."))
            except ValueError:
                pass
        ok = abs(total - 100.0) < 0.01
        self.sched_sum.configure(
            text=f"Сумма долей: {total:.0f} %  {'✓' if ok else '— должно быть 100 %'}",
            foreground=(BRASS_DEEP if ok else "#6E1423"))

    # ── Нижняя панель ──────────────────────────────────────────────────── #
    def _build_bottom_bar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom", padx=12, pady=8)

        ttk.Button(bar, text="Загрузить YAML…", command=self._load_yaml).pack(side="left")
        ttk.Button(bar, text="Сохранить YAML…", command=self._save_yaml).pack(side="left", padx=6)

        self.open_after = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Открыть PDF после сборки",
                        variable=self.open_after).pack(side="left", padx=12)

        self.go_btn = ttk.Button(bar, text="Сформировать PDF", style="Go.TButton",
                                 command=self._on_generate)
        self.go_btn.pack(side="right")

        self.status = tk.Text(self, height=4, bg=CARD, fg=INK_SOFT, relief="solid",
                              borderwidth=1, font=("Consolas", 9), wrap="word")
        self.status.pack(fill="x", side="bottom", padx=12, pady=(0, 4))
        self.status.insert("end", "Готов к работе. Заполните форму и нажмите «Сформировать PDF».\n")
        self.status.configure(state="disabled")

    def _log(self, msg: str):
        self.status.configure(state="normal")
        self.status.insert("end", msg + "\n")
        self.status.see("end")
        self.status.configure(state="disabled")
        self.update_idletasks()

    # ── Сбор и заполнение формы ───────────────────────────────────────── #
    def _set_form(self, data: dict):
        """Заполняет поля формы из «плоского» словаря DEFAULTS-формата."""
        for key, var in self.vars.items():
            if key in data and not isinstance(data[key], list):
                var.set(str(data[key]))
        # Транши.
        for _, _, row in list(self.schedule_rows):
            row.destroy()
        self.schedule_rows.clear()
        for month, pct in data.get("schedule", []):
            self._add_schedule_row(str(month), str(pct))
        self._sync_exit_label()

    def _num(self, key: str, label: str, integer: bool = False):
        raw = self.vars[key].get().strip().replace(" ", "").replace(",", ".")
        if raw == "":
            raise ConfigError(f"«{label}»: заполните поле.")
        try:
            return int(float(raw)) if integer else float(raw)
        except ValueError:
            raise ConfigError(f"«{label}»: введите число (сейчас: «{self.vars[key].get()}»).")

    def _gather(self) -> dict:
        """Собирает словарь конфига объекта из полей формы."""
        rec = self.vars["recommendation"].get().split(" ")[0] or "HOLD"
        schedule = []
        for mv, pv, _ in self.schedule_rows:
            schedule.append({"month": int(float(mv.get().replace(",", "."))),
                             "pct": float(pv.get().replace(",", "."))})

        exit_block = {"hold_years": self._num("hold_years", "Горизонт удержания", integer=True),
                      "selling_cost_pct": self._num("selling_cost_pct", "Издержки продажи")}
        exit_val = self._num("exit_value", "Значение выхода")
        if self.vars["exit_method"].get() == "growth":
            exit_block["exit_price_growth_pct"] = exit_val
            exit_var_key = "exit_price_growth_pct"
        else:
            exit_block["exit_cap_rate_pct"] = exit_val
            exit_var_key = "exit_cap_rate_pct"

        return {
            "object": {
                "name": self.vars["name"].get(), "developer": self.vars["developer"].get(),
                "zone": self.vars["zone"].get(), "type": self.vars["type"].get(),
                "area_m2": self._num("area_m2", "Площадь"),
            },
            "purchase": {
                "price_rub": self._num("price_rub", "Цена объекта"),
                "installment": {
                    "down_payment_pct": self._num("down_payment_pct", "Первый взнос"),
                    "schedule": schedule,
                },
            },
            "operations": {
                "rental_rate_rub_month": self._num("rental_rate_rub_month", "Ставка аренды"),
                "occupancy_pct": self._num("occupancy_pct", "Загрузка"),
                "opex_pct_of_revenue": self._num("opex_pct_of_revenue", "Доля OPEX"),
                "rental_growth_pct": self._num("rental_growth_pct", "Рост ставки"),
                "rent_start_month": self._num("rent_start_month", "Начало аренды", integer=True),
            },
            "exit": exit_block,
            "assumptions": {
                "discount_rate_pct": self._num("discount_rate_pct", "Ставка дисконтирования"),
                "target_irr_pct": self._num("target_irr_pct", "Целевой IRR"),
            },
            "monte_carlo": {
                "iterations": self._num("iterations", "Итераций", integer=True),
                "seed": self._num("seed", "Сид", integer=True),
                "ranges": {
                    "occupancy_pct": {"dist": "triangular",
                                      "low": self._num("occ_low", "Загрузка: мин"),
                                      "mode": self._num("occ_mode", "Загрузка: мода"),
                                      "high": self._num("occ_high", "Загрузка: макс")},
                    "rental_growth_pct": {"dist": "normal",
                                          "mean": self._num("grow_mean", "Рост: среднее"),
                                          "sd": self._num("grow_sd", "Рост: σ")},
                    exit_var_key: {"dist": "triangular",
                                   "low": self._num("exitr_low", "Выход: мин"),
                                   "mode": self._num("exitr_mode", "Выход: мода"),
                                   "high": self._num("exitr_high", "Выход: макс")},
                },
            },
            "advisor": {
                "recommendation": rec,
                "thesis": self.vars["thesis"].get(),
                "exit_scenario": self.vars["exit_scenario"].get(),
                "name": self.vars["advisor_name"].get(),
                "title": self.vars["advisor_title"].get(),
            },
        }

    # ── Кнопки ─────────────────────────────────────────────────────────── #
    def _save_yaml(self):
        try:
            data = self._gather()
        except ConfigError as e:
            messagebox.showerror("Проверьте поля", str(e))
            return
        path = filedialog.asksaveasfilename(defaultextension=".yaml",
                                            initialfile=f"{slug(data['object']['name'])}.yaml",
                                            filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            Path(path).write_text(config_to_yaml(data), encoding="utf-8")
            self._log(f"Сохранено: {path}")

    def _load_yaml(self):
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml")])
        if not path:
            return
        try:
            raw = yaml_to_dict(path)
            self._set_form(config_to_flat(raw))
            self._log(f"Загружено: {path}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Не удалось загрузить", str(e))

    def _on_generate(self):
        if self._busy:
            return
        try:
            data = self._gather()
        except ConfigError as e:
            messagebox.showerror("Проверьте поля", str(e))
            return

        out = app_dir() / "out" / f"{slug(data['object']['name'])}-memo.pdf"
        self._busy = True
        self.go_btn.configure(state="disabled", text="Сборка…")
        self._log("— Запуск сборки —")
        threading.Thread(target=self._worker, args=(data, out), daemon=True).start()

    def _worker(self, data: dict, out: Path):
        try:
            result = generate(data, out, log=lambda m: self.after(0, self._log, m))
            self.after(0, self._done_ok, result)
        except (ConfigError, Exception) as e:  # noqa: BLE001
            self.after(0, self._done_err, e)

    def _done_ok(self, result: Path):
        self._busy = False
        self.go_btn.configure(state="normal", text="Сформировать PDF")
        self._log(f"✓ Готово: {result}")
        if self.open_after.get():
            try:
                open_file(result)
            except Exception:  # noqa: BLE001
                pass
        messagebox.showinfo("Готово", f"Меморандум собран:\n{result}")

    def _done_err(self, exc: Exception):
        self._busy = False
        self.go_btn.configure(state="normal", text="Сформировать PDF")
        self._log(f"✗ Ошибка: {exc}")
        title = "Проверьте поля" if isinstance(exc, ConfigError) else "Не удалось собрать PDF"
        messagebox.showerror(title, str(exc))


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
