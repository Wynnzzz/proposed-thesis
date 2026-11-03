"""Template prompt (Inggris, karena benchmark berbahasa Inggris). Format akhir ketat: 'Final answer: X' [P11]."""

FORMAT = {
    "mmlu": "The final line must be exactly `Final answer: <X>` where <X> is a single letter A, B, C, or D.",
    "hotpotqa": "The final line must be exactly `Final answer: <X>` where <X> is a short answer "
                "(a few words, or yes/no) with no explanation.",
}

SYSTEM = ("You are a careful expert. Think step by step briefly, then give your answer. {fmt}")

ANSWER_USER = "{body}"

VERIFY_USER = ("{body}\n\nProposed answer: {answer}\n\n"
               "Is the proposed answer correct? Reply with exactly one word: Yes or No.")

REVISION_USER = ("{body}\n\nYour previous response:\n{own}\n\n"
                 "Responses from other agents in the previous round:\n{others}\n\n"
                 "Use these responses as additional information. Reconsider the question and give your updated answer. "
                 "{fmt}")
