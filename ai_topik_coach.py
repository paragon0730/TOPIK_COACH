"""
Model: EXAONE 3.5 7.8B (LG AI Research) via Ollama
- Bilingual Korean-English model, runs fully locally
- No API key, no cost, no rate limits
- Install Ollama: https://ollama.com
- Download model: ollama pull exaone3.5:7.8b
- Start server:   ollama serve
- Then run:       python ai_topik_coach.py

Requirements:
    pip install langchain langchain-ollama langchain-core

Source data:
    64th TOPIK II Reading Exam (2019), officially released by NIIED
    Available at: https://www.topikguide.com/download-64th-topik-test-papers/
"""

import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# ── 1. LLM CONFIGURATION ──────────────────────────────────────
# EXAONE 3.5 7.8B by LG AI Research — bilingual Korean-English model.
# Released December 2024, specifically optimized for Korean tasks.
# Runs locally via Ollama — no API key or internet required after setup.
# No fine-tuning applied. Korean capability is activated through
# few-shot prompting with real TOPIK II examples.
#
# Make sure Ollama is running before executing this script:
#   ollama serve        (in a separate terminal)
#   ollama pull exaone3.5:7.8b   (first time only — downloads ~4.8GB)
llm = ChatOllama(
    model="exaone3.5:7.8b",
    temperature=0.3
)

# ── 2. REAL TOPIK II QUESTIONS ────────────────────────────────
# Source: 64th TOPIK II Reading Exam (제64회 한국어능력시험 II)
# Officially released by NIIED, 2019.
# Answer keys from official NIIED score sheet.
#
# 5 questions selected to cover a range of question types:
# - Q23: 심정 파악 (Emotion Inference)       — student answer WRONG
# - Q24: 내용 일치 (Content Matching)         — student answer WRONG
# - Q32: 내용 일치 (Content Matching)         — student answer CORRECT
# - Q35: 주제 파악 (Main Idea)               — student answer WRONG
# - Q37: 주제 파악 (Main Idea)               — student answer WRONG
#
# Simulated answers: 4 wrong, 1 correct — reflects a realistic student
# profile struggling with reading comprehension, and allows the memory
# to detect a pattern of errors in main idea questions by session 5.

# Q23 and Q24 use the same passage — stored once to avoid repetition
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
        "question": "밑줄 친 부분에 나타난 '나'의 심정으로 알맞은 것을 고르십시오.\n(밑줄: '일을 하는 내내 일이 손에 잡히지 않았다')",
        "options": {
            "(1)": "걱정스럽다",
            "(2)": "불만스럽다",
            "(3)": "후회스럽다",
            "(4)": "당황스럽다"
        },
        "correct_answer": "(1)",
        "student_answer": "(4)"   # wrong
    },
    {
        "id": 24,
        "source": "64th TOPIK II Reading (2019), Q24",
        "question_type": "내용 일치 (Content Matching)",
        "passage": Q23_Q24_PASSAGE,
        "question": "위 글의 내용과 같은 것을 고르십시오.",
        "options": {
            "(1)": "그 가족은 나에게 화를 냈다.",
            "(2)": "카드 회사는 그 가족에게 연락을 했다.",
            "(3)": "나는 그 가족에게 직접 전화를 걸었다.",
            "(4)": "나는 그 가족을 찾아다니느라 일을 못 했다."
        },
        "correct_answer": "(2)",
        "student_answer": "(3)"   # wrong
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
        "options": {
            "(1)": "석주명은 한국의 나비를 총 844종으로 분류하였다.",
            "(2)": "석주명은 나비 이름을 고유어로 바꾸려고 노력하였다.",
            "(3)": "석주명은 자신의 배추흰나비 연구에 문제가 있음을 알았다.",
            "(4)": "석주명은 나비의 날개 모양이 다르면 종이 달라짐을 밝혔다."
        },
        "correct_answer": "(2)",
        "student_answer": "(2)"   # correct
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
        "options": {
            "(1)": "첨단 장비 덕분에 문화재 복원이 수월해졌다.",
            "(2)": "문화재는 손상 예방을 위한 사전 관리가 중요하다.",
            "(3)": "복원 환경 탓에 원본이 변형되는 경우가 많아지고 있다.",
            "(4)": "복원 기술자를 대상으로 한 3D 장치 사용 교육이 필요하다."
        },
        "correct_answer": "(1)",
        "student_answer": "(2)"   # wrong
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
        "options": {
            "(1)": "버섯은 다른 식물이 있어야 자랄 수 있다.",
            "(2)": "나무의 뿌리가 숲에서 하는 기능은 다양하다.",
            "(3)": "버섯은 숲에서 나무들의 정보 교환을 돕는 역할을 한다.",
            "(4)": "나무의 생활환경에 대한 학자들의 관심이 높아지고 있다."
        },
        "correct_answer": "(3)",
        "student_answer": "(1)"   # wrong
    }
]

# ── 3. FEW-SHOT EXAMPLES ─────────────────────────────────────
# Two real TOPIK II questions embedded in the prompt.
# These teach the LLM the correct question format, style, and
# difficulty level — this is called few-shot prompting.
# No model training is needed.
FEW_SHOT_EXAMPLES = """
Example 1:
Passage: 나비 박사 석주명은 나비의 종류를 분류하고 이름을 지어 준 생물학자이다.
Question type: 내용 일치
Generated question: 위 글의 내용과 같은 것을 고르십시오.
(1) 석주명은 나비 연구를 1940년대에 시작하였다.
(2) 석주명은 나비 이름을 고유어로 바꾸려고 노력하였다.
(3) 석주명은 나비의 수를 844종에서 늘렸다.
(4) 석주명은 배추흰나비만 연구하였다.
Answer: (2)

Example 2:
Passage: 문화재 복원 작업은 복원된 부분이 자연스러워야 하고 그 과정에서 문화재가 추가로 손상되지 않아야 한다.
Question type: 주제 파악
Generated question: 다음 글의 주제로 가장 알맞은 것을 고르십시오.
(1) 첨단 장비 덕분에 문화재 복원이 수월해졌다.
(2) 문화재는 손상 예방을 위한 사전 관리가 중요하다.
(3) 복원 환경 탓에 원본이 변형되는 경우가 많아지고 있다.
(4) 복원 기술자를 대상으로 한 3D 장치 사용 교육이 필요하다.
Answer: (1)
"""

# ── 4. CHAIN 1: MOCK QUESTION GENERATION ─────────────────────
# Uses few-shot prompting to generate a new TOPIK II style question.
# LCEL syntax: prompt | llm | output_parser
# The LLM learns format and difficulty from the two real examples above.
gen_template = """\
You are a TOPIK II exam question generator.
Study the following real TOPIK II question examples carefully to learn
the exact format, style, and difficulty level:

{few_shot_examples}

Now generate ONE new TOPIK II multiple-choice question of type '{question_type}'
for the following passage. Follow these rules strictly:
- Write the question in Korean
- Provide EXACTLY 4 answer options labeled (1) (2) (3) (4)
- Only one option should be correct
- The other three options should be plausible but clearly wrong
- End with: Answer: (N)

Passage:
{passage}

Generated question:"""

gen_prompt = PromptTemplate.from_template(gen_template)
generation_chain = gen_prompt | llm | StrOutputParser()

# ── 5. CHAIN 2: ANSWER EVALUATION ────────────────────────────
# Evaluates the student's answer and explains the reasoning.
# Receives chat_history so it can detect recurring error patterns.
# LCEL syntax: prompt | llm | output_parser
eval_template = """\
You are a TOPIK II exam tutor helping a non-native Korean speaker.

Previous session history (use this to detect recurring mistakes):
{chat_history}

---
Passage:
{passage}

Question: {question}

Answer options:
{options}

Correct answer: {correct_answer}
Student's answer: {student_answer}
---

Your task:
1. State clearly whether the student's answer is CORRECT or INCORRECT.
2. Explain in 2-3 sentences WHY the correct answer is right, referencing
   the passage directly.
3. If the student answered incorrectly, explain why their chosen option
   is wrong.
4. If you notice from the session history that the student has made
   similar mistakes before, point this out and give a specific tip.

Write in simple English or Korean. Be encouraging and constructive.

Feedback:"""

eval_prompt = PromptTemplate.from_template(eval_template)
eval_chain = eval_prompt | llm | StrOutputParser()

# ── 6. MANUAL CONVERSATION MEMORY ────────────────────────────
# A plain string that grows after each session.
# Passed into the evaluation prompt so the LLM can see all previous
# exchanges and detect recurring weak patterns.
# This is equivalent to ConversationBufferMemory from Lecture 7
# but implemented without any external dependency.
chat_history = ""

# ── 7. SESSION RUNNER ─────────────────────────────────────────
def run_session(q, session_id):
    """
    One full session:
    1. Generate a new mock question (Chain 1, few-shot prompting)
    2. Evaluate the student answer with explanation (Chain 2)
    3. Append the exchange to chat_history for the next session
    """
    global chat_history

    print(f"\n{'='*65}")
    print(f"Session {session_id} | Q{q['id']} | {q['question_type']}")
    print(f"Source: {q['source']}")
    print(f"{'='*65}")

    # Step 1: Generate new mock question using few-shot prompting
    print("\n[Generating new mock question in TOPIK II style...]\n")
    mock_question = generation_chain.invoke({
        "few_shot_examples": FEW_SHOT_EXAMPLES,
        "passage": q["passage"],
        "question_type": q["question_type"]
    })
    print(mock_question)

    # Format options as a readable string for the prompt
    options_str = "\n".join([f"{k}: {v}" for k, v in q["options"].items()])

    # Step 2: Evaluate student answer using accumulated session history
    print(f"\n[Real Question]\n{q['question']}")
    print(f"\n[Options]\n{options_str}")
    print(f"\n[Student Answer] {q['student_answer']}  |  [Correct Answer] {q['correct_answer']}")

    feedback = eval_chain.invoke({
        "chat_history": chat_history,
        "passage": q["passage"],
        "question": q["question"],
        "options": options_str,
        "correct_answer": q["correct_answer"],
        "student_answer": q["student_answer"]
    })
    print(f"\n[Feedback]\n{feedback}")

    # Step 3: Append this session to memory
    # The next session will see this exchange and detect patterns
    chat_history += (
        f"\nSession {session_id} | {q['question_type']}\n"
        f"Question: {q['question']}\n"
        f"Student answered: {q['student_answer']} "
        f"(Correct: {q['correct_answer']})\n"
        f"Feedback: {feedback}\n"
        f"{'-'*40}\n"
    )

    is_correct = q["student_answer"] == q["correct_answer"]
    return is_correct, mock_question, feedback

# ── 8. MAIN EXPERIMENT LOOP ───────────────────────────────────
if __name__ == "__main__":
    results = []

    for i, q in enumerate(QUESTIONS, start=1):
        try:
            is_correct, mock_q, feedback = run_session(q, session_id=i)
            results.append({
                "session": i,
                "question_id": q["id"],
                "question_type": q["question_type"],
                "student_answer": q["student_answer"],
                "correct_answer": q["correct_answer"],
                "is_correct": is_correct,
                "mock_question": mock_q,
                "feedback": feedback
            })

        except Exception as e:
            print(f"\n[ERROR in Session {i}] {e}")
            print("Make sure Ollama is running: ollama serve")
            print("Make sure model is downloaded: ollama pull exaone3.5:7.8b")
            break

    # Final summary
    correct_count = sum(1 for r in results if r["is_correct"])
    print(f"\n{'='*65}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Sessions completed : {len(results)} / {len(QUESTIONS)}")
    print(f"Student correct    : {correct_count} / {len(results)}")
    print(f"{'='*65}")

    # Save full results to JSON for manual evaluation
    with open("topik_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Full results saved to: topik_results.json")

    # Print accumulated memory to verify cross-session tracking
    print("\n[Accumulated Chat History]\n")
    print(chat_history)