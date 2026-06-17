"""
AI TOPIK Coach  —  v2 (Assignment 2)
=====================================
Model: EXAONE 3.5 7.8B (LG AI Research) via Ollama
- Bilingual Korean-English model, runs fully locally
- No API key, no cost, no rate limits
- Install Ollama:  https://ollama.com
- Download model:  ollama pull exaone3.5:7.8b
- Start server:    ollama serve
- Then run:        python ai_topik_coach_v2.py

Requirements:
    pip install langchain langchain-ollama langchain-core matplotlib

Source data:
    64th TOPIK II Reading Exam (2019), officially released by NIIED
    https://www.topikguide.com/download-64th-topik-test-papers/

------------------------------------------------------------------------
WHAT'S NEW IN v2  —  three related works applied to three components
------------------------------------------------------------------------
[Paper 1]  Feng, W., Lee, J., McNichols, H., Scarlatos, A., Smith, D.,
           Woodhead, S., Ornelas, N. O., & Lan, A. (2024). Exploring
           Automated Distractor Generation for Math Multiple-choice
           Questions via Large Language Models. Findings of NAACL 2024,
           3067-3082. arXiv:2404.02124.
   FINDING: LLMs generate valid distractors but fail to anticipate the
            misconceptions real students fall for; the fix is in-context
            learning that ties each distractor to a NAMED misconception.
   APPLIED TO: Chain 1 (question generation). We add a per-type
            misconception taxonomy and require every distractor to target
            a specific misconception -> diagnostic, not random, options.

[Paper 2]  Dhuliawala, S., Komeili, M., Xu, J., Raileanu, R., Li, X.,
           Celikyilmaz, A., & Weston, J. (2024). Chain-of-Verification
           Reduces Hallucination in Large Language Models. Findings of
           ACL 2024, 3563-3578. arXiv:2309.11495.
   FINDING: draft -> plan verification questions -> answer them
            independently against the source -> revise, cuts hallucination.
   APPLIED TO: Chain 2 (feedback). Splits evaluation into a DRAFT pass and
            a VERIFY pass that checks each claim against a quoted passage
            sentence and removes anything unsupported. Fixes the v1
            "overinterpretation" limitation.

[Paper 3]  Scarlatos, A., Smith, D., Woodhead, S., & Lan, A. (2024).
           Exploring Knowledge Tracing in Tutor-Student Dialogues using
           LLMs. arXiv:2409.16490.
   FINDING: tag the knowledge component per turn and track mastery over
            the dialogue instead of dumping the raw transcript.
   APPLIED TO: Memory. Replaces the unbounded chat_history string with a
            structured per-skill student model (attempts, correctness, the
            misconception fallen for, a mastery estimate). Fixes BOTH the
            weak pattern detection AND the v1 context-window-growth
            limitation: the eval prompt now sees a compact, bounded profile.
------------------------------------------------------------------------
"""

import os
import csv
import json

# matplotlib is only needed for the Results chart; import lazily so the
# core experiment still runs on a machine without it.
try:
    import matplotlib
    matplotlib.use("Agg")          # headless: write PNG, no display needed
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ── 1. LLM CONFIGURATION ──────────────────────────────────────
# EXAONE 3.5 7.8B by LG AI Research, run locally via Ollama.
# No fine-tuning. Korean ability activated through few-shot prompting.
#
# MOCK_MODE lets you develop the pipeline (knowledge tracing, verification
# orchestration, plotting) WITHOUT a running Ollama server — useful on
# Windows when you just want to test the plumbing. Set to False (default
# auto-detect) for the real experiment.
MODEL_NAME  = "exaone3.5:7.8b"
TEMPERATURE = 0.3
MOCK_MODE   = os.environ.get("TOPIK_MOCK", "0") == "1"

if not MOCK_MODE:
    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)

    def make_chain(template: str):
        """Build an LCEL chain: prompt | llm | parser."""
        return PromptTemplate.from_template(template) | llm | StrOutputParser()
else:
    # Minimal mock so the structured logic is testable without Ollama.
    import re

    def make_chain(template: str):
        def _invoke(d):
            t = template
            if "Generated question" in template:          # Chain 1
                return ("**질문:** (mock) 위 글의 내용으로 알맞은 것은?\n"
                        "(1) mock-correct\n(2) mock-distractor-A\n"
                        "(3) mock-distractor-B\n(4) mock-distractor-C\n"
                        "Answer: (1)\n"
                        "[Misconception map] (2)=surface-detail "
                        "(3)=opposite-polarity (4)=over-generalisation")
            if "verification questions" in template:        # Chain 2 verify
                return ("[Verified feedback] (mock) Every claim below is "
                        "grounded in a quoted passage sentence. "
                        "Result: INCORRECT.")
            return ("**1. Result:** INCORRECT\n"
                    "**2. Why correct answer is right:** (mock draft) ...\n"
                    "**3. Why your choice is wrong:** (mock draft) ...")
        class _C:
            def invoke(self, d): return _invoke(d)
        return _C()


# ── 2. KNOWLEDGE COMPONENTS + MISCONCEPTION TAXONOMY ──────────
# [Paper 1 + Paper 3] Each TOPIK II reading question type is treated as a
# knowledge component (KC). For each KC we enumerate the misconceptions
# self-studying learners most commonly fall for. These are injected into
# the generator (so distractors are diagnostic) and reused as the label
# space for knowledge tracing (so we record WHICH misconception was hit).
MISCONCEPTIONS = {
    "심정 파악 (Emotion Inference)": [
        "momentary-reaction: picks an immediate reaction word "
        "(surprise/embarrassment) instead of the sustained feeling the passage builds",
        "polarity-flip: confuses a negative feeling for its opposite",
        "literal-word-match: chooses an emotion word that literally appears "
        "near the underline but isn't what the narrator feels",
    ],
    "내용 일치 (Content Matching)": [
        "agent-swap: confuses who did the action (subject of a verb)",
        "unstated-detail: selects a plausible real-world fact the passage "
        "never actually states",
        "time-order: misreads the sequence of events (before vs after)",
    ],
    "주제 파악 (Main Idea)": [
        "peripheral-over-central: picks a true supporting detail instead of "
        "the central claim (the v1 student's dominant error)",
        "over-generalisation: broadens the topic beyond what the passage argues",
        "example-as-thesis: mistakes a single example for the main point",
    ],
}


# ── 3. REAL TOPIK II QUESTIONS ────────────────────────────────
# Source: 64th TOPIK II Reading Exam (제64회 한국어능력시험 II), NIIED 2019.
# Answer keys from the official NIIED score sheet.
Q23_Q24_PASSAGE = """놀이공원 매표소에서 아르바이트를 했다. 아르바이트가 처음이라 실수를
하지 않으려고 늘 긴장하면서 일을 했다. 어느 날, 놀러 온 한 가족에게
인원수만큼 표를 줬다. 그런데 그 가족을 보내고 나서 이용권 한 장의 값이
더 결제된 것을 알아차렸다. 바로 카드사로 전화해 고객의 전화번호를
물었지만 상담원은 알려 줄 수 없다고 했다. 하지만 내 연락처를 고객에게
전달해 주겠다고 했다. 일을 하는 내내 일이 손에 잡히지 않았다. 퇴근
시간 무렵 드디어 그 가족에게서 전화가 왔다. 내가 한 실수에 화를 낼지도
모른다는 생각에 떨리는 목소리로 상황을 설명하자 그 가족은 '놀이 기구를
타고 노느라 문자 메시지가 온 줄 몰랐어요. 많이 기다렸겠어요.'라고 하며
따뜻하게 말해 주었다."""

QUESTIONS = [
    {
        "id": 23,
        "source": "64th TOPIK II Reading (2019), Q23",
        "question_type": "심정 파악 (Emotion Inference)",
        "passage": Q23_Q24_PASSAGE,
        "question": "밑줄 친 부분에 나타난 '나'의 심정으로 알맞은 것을 고르십시오.\n"
                    "(밑줄: '일을 하는 내내 일이 손에 잡히지 않았다')",
        "options": {"(1)": "걱정스럽다", "(2)": "불만스럽다",
                    "(3)": "후회스럽다", "(4)": "당황스럽다"},
        "correct_answer": "(1)",
        "student_answer": "(4)",                       # wrong
        # which misconception this wrong pick reflects (for knowledge tracing)
        "student_misconception": "momentary-reaction",
    },
    {
        "id": 24,
        "source": "64th TOPIK II Reading (2019), Q24",
        "question_type": "내용 일치 (Content Matching)",
        "passage": Q23_Q24_PASSAGE,
        "question": "위 글의 내용과 같은 것을 고르십시오.",
        "options": {"(1)": "그 가족은 나에게 화를 냈다.",
                    "(2)": "카드 회사는 그 가족에게 연락을 했다.",
                    "(3)": "나는 그 가족에게 직접 전화를 걸었다.",
                    "(4)": "나는 그 가족을 찾아다니느라 일을 못 했다."},
        "correct_answer": "(2)",
        "student_answer": "(3)",                       # wrong
        "student_misconception": "agent-swap",
    },
    {
        "id": 32,
        "source": "64th TOPIK II Reading (2019), Q32",
        "question_type": "내용 일치 (Content Matching)",
        "passage": """나비 박사 석주명은 나비의 종류를 분류하고 이름을 지어 준 생물학자이다.
1931년부터 나비를 연구한 그는 한국의 나비가 총 844종이라는 당시의
분류를 248종으로 수정하였다. 날개 무늬나 모양이 조금만 달라도 다른
종이라고 판단한 기존의 분류가 틀렸음을 배추흰나비 16만여 마리의 무늬를
비교해서 밝혔다. 또한 그때까지 한자어나 외래어로 명명된 나비에 '떠들썩
팔랑나비'와 같은 고유어 이름을 지어 주는 데 앞장섰다.""",
        "question": "위 글의 내용과 같은 것을 고르십시오.",
        "options": {"(1)": "석주명은 한국의 나비를 총 844종으로 분류하였다.",
                    "(2)": "석주명은 나비 이름을 고유어로 바꾸려고 노력하였다.",
                    "(3)": "석주명은 자신의 배추흰나비 연구에 문제가 있음을 알았다.",
                    "(4)": "석주명은 나비의 날개 모양이 다르면 종이 달라짐을 밝혔다."},
        "correct_answer": "(2)",
        "student_answer": "(2)",                       # correct
        "student_misconception": None,
    },
    {
        "id": 35,
        "source": "64th TOPIK II Reading (2019), Q35",
        "question_type": "주제 파악 (Main Idea)",
        "passage": """문화재 복원 작업은 복원된 부분이 자연스러워야 하고 그 과정에서
문화재가 추가로 손상되지 않아야 한다. 이 때문에 정확한 측정으로 복원할
부분을 원래 모습과 동일하게 만들어 내는 것은 복원의 성공을 결정하는
중요한 요건이다. 최근 3D 스캐너와 프린터가 등장하여 이러한 요건을
충족할 수 있게 되면서 정밀하고 안전한 문화재 복원이 가능해졌다.""",
        "question": "다음 글의 주제로 가장 알맞은 것을 고르십시오.",
        "options": {"(1)": "첨단 장비 덕분에 문화재 복원이 수월해졌다.",
                    "(2)": "문화재는 손상 예방을 위한 사전 관리가 중요하다.",
                    "(3)": "복원 환경 탓에 원본이 변형되는 경우가 많아지고 있다.",
                    "(4)": "복원 기술자를 대상으로 한 3D 장치 사용 교육이 필요하다."},
        "correct_answer": "(1)",
        "student_answer": "(2)",                       # wrong
        "student_misconception": "peripheral-over-central",
    },
    {
        "id": 37,
        "source": "64th TOPIK II Reading (2019), Q37",
        "question_type": "주제 파악 (Main Idea)",
        "passage": """나무에 붙어 자라는 버섯을 보면 나무로부터 양분을 받으며 별다른 노력
없이 살아간다고 생각하기 쉽다. 하지만 버섯은 나무에게 없어서는 안 될
중요한 존재이다. 나무들은 위기 상황이 발생해도 자리를 옮겨 이를 알릴
수 없기 때문에 뿌리로 소통하며 위험에 대비한다. 이때 뿌리가 짧아 서로
닿지 않는 나무들 사이에서는 실처럼 뻗은 버섯 균사체가 메시지 전달을
대신한다. 그래서 학자들은 버섯 균류를 '숲의 통신망'이라고 부른다.""",
        "question": "다음 글의 주제로 가장 알맞은 것을 고르십시오.",
        "options": {"(1)": "버섯은 다른 식물이 있어야 자랄 수 있다.",
                    "(2)": "나무의 뿌리가 숲에서 하는 기능은 다양하다.",
                    "(3)": "버섯은 숲에서 나무들의 정보 교환을 돕는 역할을 한다.",
                    "(4)": "나무의 생활환경에 대한 학자들의 관심이 높아지고 있다."},
        "correct_answer": "(3)",
        "student_answer": "(1)",                       # wrong
        "student_misconception": "peripheral-over-central",
    },
]


# ── 4. FEW-SHOT EXAMPLES (now misconception-annotated) ────────
# [Paper 1] Each demonstration labels which misconception each distractor
# targets, teaching the model to make options diagnostic, not just wrong.
FEW_SHOT_EXAMPLES = """
Example 1:
Passage: 나비 박사 석주명은 나비의 종류를 분류하고 이름을 지어 준 생물학자이다.
Question type: 내용 일치
Generated question: 위 글의 내용과 같은 것을 고르십시오.
(1) 석주명은 나비 이름을 고유어로 바꾸려고 노력하였다.   -> correct
(2) 석주명은 나비 연구를 1940년대에 시작하였다.          -> unstated-detail
(3) 석주명은 나비의 수를 844종에서 늘렸다.               -> time-order / direction flip
(4) 석주명은 배추흰나비만 연구하였다.                    -> over-generalisation
Answer: (1)

Example 2:
Passage: 문화재 복원 작업은 복원된 부분이 자연스러워야 하고 그 과정에서 문화재가 추가로 손상되지 않아야 한다.
Question type: 주제 파악
Generated question: 다음 글의 주제로 가장 알맞은 것을 고르십시오.
(1) 첨단 장비 덕분에 문화재 복원이 수월해졌다.            -> correct
(2) 문화재는 손상 예방을 위한 사전 관리가 중요하다.        -> peripheral-over-central
(3) 복원 환경 탓에 원본이 변형되는 경우가 많아지고 있다.    -> polarity-flip
(4) 복원 기술자를 대상으로 한 3D 장치 사용 교육이 필요하다. -> example-as-thesis
Answer: (1)
"""


# ── 5. CHAIN 1: MISCONCEPTION-AWARE QUESTION GENERATION ───────
# [Paper 1] The generator must tie each distractor to a named misconception
# drawn from the taxonomy for this question type.
gen_template = """\
You are a TOPIK II exam question generator.
Study these real, misconception-annotated examples to learn the exact
format, style, and difficulty:

{few_shot_examples}

For the question type '{question_type}', target ONE of these common
learner misconceptions per distractor:
{misconception_list}

Now generate ONE new TOPIK II multiple-choice question for the passage below.
Rules:
- Write the question in Korean.
- Provide EXACTLY 4 options labeled (1) (2) (3) (4); exactly one is correct.
- Each of the three distractors must target a DIFFERENT misconception above
  and be plausible to a learner who holds that misconception.
- End with: Answer: (N)
- Then on a final line, output the mapping, e.g.:
  [Misconception map] (2)=agent-swap (3)=unstated-detail (4)=time-order

Passage:
{passage}

Generated question:"""

generation_chain = make_chain(gen_template)


# ── 6. CHAIN 2: ANSWER EVALUATION (Chain-of-Verification) ─────
# [Paper 2] Two passes. DRAFT writes the feedback; VERIFY interrogates each
# claim against the passage and rewrites, dropping anything unsupported.
draft_template = """\
You are a TOPIK II reading tutor for a non-native Korean speaker.

Student profile (recurring strengths/weaknesses so far):
{student_profile}

---
Passage:
{passage}

Question: {question}

Options:
{options}

Correct answer: {correct_answer}
Student's answer: {student_answer}
---

Write DRAFT feedback:
1. State clearly CORRECT or INCORRECT.
2. In 2-3 sentences, explain WHY the correct answer is right, referencing
   the passage.
3. If wrong, explain why the chosen option is wrong.
4. If the profile shows a recurring weakness relevant here, name it and give
   one concrete tip.

Draft feedback:"""

draft_chain = make_chain(draft_template)

verify_template = """\
You are a strict fact-checker reviewing a tutor's DRAFT feedback against the
source passage. Your job is to remove any claim the passage does not support.

Passage:
{passage}

DRAFT feedback to check:
{draft_feedback}

Step 1 - Plan verification questions: for every factual claim the draft makes
about the passage, write a question of the form "Does the passage state X?".
Step 2 - Answer each independently. Quote the exact passage sentence that
supports it. If no sentence supports it, mark the claim UNSUPPORTED.
Step 3 - Rewrite the feedback keeping ONLY claims backed by a quoted sentence.
Remove or soften every UNSUPPORTED claim. Keep the CORRECT/INCORRECT verdict,
keep it encouraging, and keep it concise.

Output ONLY the final verified feedback (no verification questions):"""

verify_chain = make_chain(verify_template)


# ── 7. STRUCTURED KNOWLEDGE-TRACING MEMORY ───────────────────
# [Paper 3] Replaces the v1 unbounded chat_history string. We keep a compact
# per-KC (per question-type) state and render a bounded profile for the prompt.
student_model = {}   # question_type -> dict(attempts, correct, mastery, misconceptions{})

def _mastery(correct, attempts):
    # Laplace-smoothed mastery so a single attempt isn't 0% or 100%.
    return round((correct + 1) / (attempts + 2), 2)

def update_student_model(q, is_correct):
    """Update the knowledge-tracing state after a session."""
    kc = q["question_type"]
    s = student_model.setdefault(
        kc, {"attempts": 0, "correct": 0, "mastery": 0.0, "misconceptions": {}})
    s["attempts"] += 1
    if is_correct:
        s["correct"] += 1
    else:
        m = q.get("student_misconception")
        if m:
            s["misconceptions"][m] = s["misconceptions"].get(m, 0) + 1
    s["mastery"] = _mastery(s["correct"], s["attempts"])
    return s["mastery"]

def render_student_profile():
    """Compact, BOUNDED profile string for the eval prompt (fixes the v1
    context-growth limitation: size is O(#question types), not O(#sessions))."""
    if not student_model:
        return "No history yet — first session."
    lines = []
    for kc, s in student_model.items():
        top = max(s["misconceptions"], key=s["misconceptions"].get) \
              if s["misconceptions"] else "none"
        lines.append(
            f"- {kc}: {s['correct']}/{s['attempts']} correct "
            f"(mastery {s['mastery']:.2f}); recurring error: {top}")
    return "\n".join(lines)


# ── 8. SESSION RUNNER ─────────────────────────────────────────
def run_session(q, session_id):
    print(f"\n{'='*65}")
    print(f"Session {session_id} | Q{q['id']} | {q['question_type']}")
    print(f"Source: {q['source']}")
    print(f"{'='*65}")

    # Step 1 — generate a misconception-aware mock question  [Paper 1]
    print("\n[Generating misconception-targeted mock question...]\n")
    misconception_list = "\n".join(
        f"  - {m}" for m in MISCONCEPTIONS[q["question_type"]])
    mock_question = generation_chain.invoke({
        "few_shot_examples": FEW_SHOT_EXAMPLES,
        "passage": q["passage"],
        "question_type": q["question_type"],
        "misconception_list": misconception_list,
    })
    print(mock_question)

    options_str = "\n".join(f"{k}: {v}" for k, v in q["options"].items())
    profile = render_student_profile()                       # [Paper 3]

    print(f"\n[Real Question]\n{q['question']}")
    print(f"\n[Options]\n{options_str}")
    print(f"\n[Student Answer] {q['student_answer']}  |  "
          f"[Correct Answer] {q['correct_answer']}")

    # Step 2 — DRAFT feedback                                  [Paper 2]
    draft = draft_chain.invoke({
        "student_profile": profile,
        "passage": q["passage"],
        "question": q["question"],
        "options": options_str,
        "correct_answer": q["correct_answer"],
        "student_answer": q["student_answer"],
    })

    # Step 3 — VERIFY feedback against the passage             [Paper 2]
    feedback = verify_chain.invoke({
        "passage": q["passage"],
        "draft_feedback": draft,
    })
    print(f"\n[Verified Feedback]\n{feedback}")

    # Step 4 — update knowledge-tracing state                  [Paper 3]
    is_correct = q["student_answer"] == q["correct_answer"]
    mastery_after = update_student_model(q, is_correct)
    print(f"\n[Knowledge state] {q['question_type']} mastery -> {mastery_after}")

    return {
        "session": session_id,
        "question_id": q["id"],
        "question_type": q["question_type"],
        "student_answer": q["student_answer"],
        "correct_answer": q["correct_answer"],
        "is_correct": is_correct,
        "misconception": q.get("student_misconception"),
        "mastery_after": mastery_after,
        "mock_question": mock_question,
        "draft_feedback": draft,
        "verified_feedback": feedback,
        # snapshot of every KC's mastery after this session (for the chart)
        "mastery_snapshot": {kc: s["mastery"] for kc, s in student_model.items()},
    }


# ── 9. RESULTS: TABLE + CHART ────────────────────────────────
def write_summary_csv(results, path="per_type_summary.csv"):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question_type", "attempts", "correct",
                    "accuracy", "final_mastery", "top_misconception"])
        for kc, s in student_model.items():
            top = max(s["misconceptions"], key=s["misconceptions"].get) \
                  if s["misconceptions"] else "none"
            acc = round(s["correct"] / s["attempts"], 2) if s["attempts"] else 0
            w.writerow([kc, s["attempts"], s["correct"], acc, s["mastery"], top])
    print(f"Per-type summary table saved to: {path}")

def plot_mastery(results, path="mastery_over_sessions.png"):
    if not HAS_MPL:
        print("matplotlib not installed — skipping chart "
              "(pip install matplotlib).")
        return
    types = list(student_model.keys())
    short = {                       # ASCII labels avoid Korean-font issues
        "심정 파악 (Emotion Inference)": "Emotion Inference",
        "내용 일치 (Content Matching)": "Content Matching",
        "주제 파악 (Main Idea)": "Main Idea",
    }
    sessions = [r["session"] for r in results]
    plt.figure(figsize=(8, 5))
    for kc in types:
        ys = []
        last = None
        for r in results:
            last = r["mastery_snapshot"].get(kc, last)
            ys.append(last)
        plt.plot(sessions, ys, marker="o", label=short.get(kc, kc))
    plt.xlabel("Session")
    plt.ylabel("Mastery (Laplace-smoothed)")
    plt.title("Knowledge tracing: per-type mastery across sessions")
    plt.ylim(0, 1)
    plt.xticks(sessions)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Mastery chart saved to: {path}")


# ── 10. MAIN EXPERIMENT LOOP ─────────────────────────────────
if __name__ == "__main__":
    results = []
    for i, q in enumerate(QUESTIONS, start=1):
        try:
            results.append(run_session(q, session_id=i))
        except Exception as e:
            print(f"\n[ERROR in Session {i}] {e}")
            print("Make sure Ollama is running:  ollama serve")
            print("Make sure model is pulled:    ollama pull exaone3.5:7.8b")
            break

    correct_count = sum(1 for r in results if r["is_correct"])
    print(f"\n{'='*65}")
    print("EXPERIMENT COMPLETE")
    print(f"Sessions completed : {len(results)} / {len(QUESTIONS)}")
    print(f"Student correct    : {correct_count} / {len(results)}")
    print(f"{'='*65}")

    with open("topik_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Full results saved to: topik_results.json")

    write_summary_csv(results)
    plot_mastery(results)

    print("\n[Final knowledge-tracing state]")
    print(render_student_profile())
