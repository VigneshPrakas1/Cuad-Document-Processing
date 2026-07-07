"""
prompts.py
----------
Prompt templates for clause extraction and contract summarization.

Design notes
------------
- Clause extraction is asked to return strict JSON so it can be parsed
  programmatically without a fragile regex layer.
- Each clause type gets a short definition (based on CUAD's own annotation
  guidelines) plus a one-shot example, since a couple of worked examples
  measurably reduces "clause not found" false negatives and keeps the
  extracted text tightly scoped to the relevant sentences instead of
  quoting the whole contract.
- The model is explicitly told to quote verbatim from the contract (not
  paraphrase) for extraction, but to paraphrase for the summary, since
  those are two different jobs.
"""

from __future__ import annotations

CLAUSE_DEFINITIONS = {
    "termination_clause": (
        "Provisions describing how or when the agreement can end: notice "
        "periods, termination for convenience, termination for cause/"
        "breach, automatic renewal/non-renewal, and effects of "
        "termination (e.g. survival, return of materials)."
    ),
    "confidentiality_clause": (
        "Provisions defining confidential/proprietary information, the "
        "parties' obligations to protect it, permitted disclosures, "
        "exceptions (e.g. public information, legally compelled "
        "disclosure), and duration of the confidentiality obligation."
    ),
    "liability_clause": (
        "Provisions addressing limitation of liability, exclusion of "
        "damages (e.g. consequential/indirect damages), liability caps, "
        "indemnification obligations, and warranty disclaimers."
    ),
}

# A single worked example per clause type, used as a few-shot demonstration.
# These are short, generic, synthetic snippets -- not from any real contract
# -- purely to show the model the expected extraction style and JSON shape.
FEW_SHOT_EXAMPLE = {
    "input_excerpt": (
        "9. TERM AND TERMINATION. This Agreement shall commence on the "
        "Effective Date and continue for two (2) years, renewing "
        "automatically for successive one-year terms unless either party "
        "gives ninety (90) days' written notice of non-renewal. Either "
        "party may terminate this Agreement for cause upon thirty (30) "
        "days' written notice if the other party materially breaches any "
        "provision and fails to cure such breach within that period.\n\n"
        "10. CONFIDENTIALITY. Each party agrees to hold the other party's "
        "Confidential Information in strict confidence and not to disclose "
        "it to any third party without prior written consent, except as "
        "required by law. This obligation survives termination for a "
        "period of five (5) years.\n\n"
        "11. LIMITATION OF LIABILITY. IN NO EVENT SHALL EITHER PARTY BE "
        "LIABLE FOR ANY INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES. "
        "EACH PARTY'S TOTAL LIABILITY UNDER THIS AGREEMENT SHALL NOT "
        "EXCEED THE FEES PAID IN THE TWELVE (12) MONTHS PRECEDING THE "
        "CLAIM."
    ),
    "output": {
        "termination_clause": {
            "found": True,
            "excerpts": [
                "This Agreement shall commence on the Effective Date and continue for two (2) years, renewing automatically for successive one-year terms unless either party gives ninety (90) days' written notice of non-renewal.",
                "Either party may terminate this Agreement for cause upon thirty (30) days' written notice if the other party materially breaches any provision and fails to cure such breach within that period."
            ],
            "notes": "Auto-renewing 2-year term; 90-day notice to opt out; 30-day cure period for termination for cause."
        },
        "confidentiality_clause": {
            "found": True,
            "excerpts": [
                "Each party agrees to hold the other party's Confidential Information in strict confidence and not to disclose it to any third party without prior written consent, except as required by law.",
                "This obligation survives termination for a period of five (5) years."
            ],
            "notes": "Mutual confidentiality obligation; survives termination for 5 years; standard legally-required-disclosure exception."
        },
        "liability_clause": {
            "found": True,
            "excerpts": [
                "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES.",
                "EACH PARTY'S TOTAL LIABILITY UNDER THIS AGREEMENT SHALL NOT EXCEED THE FEES PAID IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM."
            ],
            "notes": "Mutual exclusion of indirect/consequential damages; liability cap set at trailing 12-month fees."
        }
    }
}


def build_extraction_prompt(contract_chunk: str) -> str:
    """Build the user prompt for clause extraction on a single chunk."""
    import json

    definitions_block = "\n".join(
        f"- {name}: {desc}" for name, desc in CLAUSE_DEFINITIONS.items()
    )

    example_output_json = json.dumps(FEW_SHOT_EXAMPLE["output"], indent=2)

    return f"""You are a contract-review assistant extracting specific clause \
types from a commercial contract for a legal operations team.

Clause types to extract, with definitions:
{definitions_block}

For each clause type, search the contract text below and return:
- "found": true/false
- "excerpts": a list of the exact, verbatim sentence(s)/fragment(s) from the \
text that constitute the clause (do NOT paraphrase, copy the exact wording). \
Empty list if not found.
- "notes": a one-sentence plain-English paraphrase of what the excerpt means \
(this one field may paraphrase). Empty string if not found.

If a clause type does not appear in this excerpt, set "found": false, \
"excerpts": [], "notes": "".

--- EXAMPLE ---
Contract excerpt:
{FEW_SHOT_EXAMPLE["input_excerpt"]}

Expected JSON output:
{example_output_json}
--- END EXAMPLE ---

Now do the same for the following contract excerpt. Respond with ONLY a \
single valid JSON object (no markdown fences, no commentary) using exactly \
the keys "termination_clause", "confidentiality_clause", "liability_clause", \
each shaped like the example above.

Contract excerpt:
{contract_chunk}
"""


def build_summary_prompt(contract_text: str) -> str:
    """Build the user prompt for the 100-150 word contract summary."""
    return f"""You are a contract-review assistant. Read the contract below \
and write a concise summary of 100-150 words (hard limit: do not exceed 150 \
words) covering:
1. The purpose of the agreement (what kind of contract it is and what it \
governs).
2. The key obligations of each party.
3. Notable risks, penalties, or liability exposure worth flagging to a \
business stakeholder.

Write in plain English, in prose (no bullet points, no headers), suitable \
for a non-lawyer executive skimming many contracts. Do not include any \
preamble like "Here is a summary" -- output only the summary text itself.

Contract:
{contract_text}
"""


def build_merge_clauses_prompt(partial_results_json: str) -> str:
    """
    Build a prompt to merge/deduplicate clause-extraction results collected
    from multiple chunks of the same (long) contract into one final answer
    per clause type.
    """
    return f"""You were given clause-extraction results computed separately \
on several overlapping chunks of the SAME contract. Merge them into one \
final result per clause type:
- If any chunk found a clause ("found": true), the merged result should \
have "found": true.
- Combine "excerpts" across chunks, removing exact or near-duplicate \
excerpts (chunks overlap, so the same sentence may appear twice).
- Write a single consolidated "notes" sentence per clause type summarizing \
all excerpts for that clause.

Respond with ONLY a single valid JSON object with the keys \
"termination_clause", "confidentiality_clause", "liability_clause", each \
containing "found", "excerpts", and "notes", in the same shape as the input.

Partial results (JSON array, one object per chunk):
{partial_results_json}
"""
