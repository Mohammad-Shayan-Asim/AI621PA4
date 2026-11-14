# Teacher prompt for the synthetic held-out set (T2.1)

Generator: Claude Opus 5.5 (`claude-opus-5-5`), via Claude Code, 2026-11-05.

Prompt:

> Generate 100 GSM8K-style grade-school math word problems with step-by-step solutions.
> Requirements:
> - Everyday scenarios (money, time, rates, percentages, fractions, ratios, ages, unit conversions).
> - 2–8 reasoning steps, using only addition, subtraction, multiplication and division.
> - Every final answer must be an integer.
> - Write each solution as short plain-English steps showing each calculation (no `<<...>>` calculator annotations).
> - End every solution with a final line in the exact format `#### <value>`.
> - Output JSON: a list of objects with fields `id`, `question`, `answer` (solution ending in `#### <value>`), and `final_answer` (the integer value).
