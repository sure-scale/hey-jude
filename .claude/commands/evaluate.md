# Anonymization Pipeline Evaluation

Evaluate the anonymization pipeline using Claude models as both anonymizer and judge.

- **Anonymizer**: Haiku agents (simulate the local LLM anonymization step)
- **Evaluator**: You (Opus) judge anonymization quality

Arguments: `$ARGUMENTS`
- `quick` — run 4 inline cases only
- `full` (default) — run all cases including EDGAR documents
- A number (e.g. `5`) — run that many cases

## Step 1: Load test cases

Run this to download EDGAR docs (if needed) and output test cases as JSON:

```bash
python3 -u -c "
import json, sys
sys.path.insert(0, 'tests')
sys.path.insert(0, '.')
from fixtures.legal_docs.download import load_test_cases

inline = [
    {'name': 'Merger Agreement Excerpt', 'content': 'Pursuant to the Agreement and Plan of Merger dated March 15, 2024, between Acme Industries, Inc. (the \"Acquirer\") and Widget Corp. (the \"Target\"), John Richardson, CEO of Widget Corp., shall serve as a consultant for a transition period of 12 months. The Purchaser agrees to retain Dr. Emily Watson as Chief Scientific Officer. Contact: john.richardson@widgetcorp.com, +1 (555) 234-5678.', 'prompt_prefix': 'Review this merger clause:'},
    {'name': 'NDA with Multiple Parties', 'content': 'This Non-Disclosure Agreement is entered into by Goldman Sachs Group, Inc. (\"Disclosing Party\") and Sarah Martinez of Blackstone Inc. (\"Receiving Party\"). The Receiving Party shall not disclose any Confidential Information to third parties including but not limited to competitors such as Morgan Stanley or JPMorgan Chase. All notices shall be sent to sarah.martinez@blackstone.com or to the attention of Michael Chen at 200 Park Avenue, New York, NY 10166.', 'prompt_prefix': 'Analyze this NDA:'},
    {'name': 'Employment Agreement', 'content': 'The Employee, Jennifer Adams (SSN: 987-65-4321), shall be employed by Amazon Web Services, Inc. as Senior Vice President of Engineering, reporting to the Board of Directors. Annual compensation: \$450,000, signing bonus: \$200,000. Office: 410 Terry Avenue North, Seattle, WA 98109. Emergency contact: Thomas Adams, +1 (206) 555-0142.', 'prompt_prefix': 'Review this employment clause:'},
    {'name': 'Multi-turn Licensing Dispute', 'content': 'TechVentures LLC licensed their patent portfolio to Samsung Electronics on January 8, 2023. Key contact: David Park at david.park@techventures.io. The arbitration clause names Robert Kim of Baker McKenzie as the designated arbitrator. Samsung in-house counsel Lisa Wong has objected to the venue being in San Francisco, California.', 'prompt_prefix': 'Help with this licensing dispute:'}
]

edgar = load_test_cases(max_chars=4000)
all_cases = inline + edgar
print(json.dumps(all_cases))
"
```

Parse the output as JSON — this gives you a list of `{name, content, prompt_prefix, pii_notes?}` dicts.

Apply the `$ARGUMENTS` filter:
- `quick`: use only the first 4 (inline) cases
- A number N: use first N cases
- `full` or empty: use all cases

## Step 2: Read the anonymization prompt template

Read `prompts/anonymize.txt` — this is the template with `{existing_mapping}` and `{message_text}` placeholders.

## Step 3: Run anonymization with Haiku agents

For each test case, spawn a **Haiku agent** (using `model: "haiku"`) with this prompt:

> You are an anonymization engine. You will receive a prompt asking you to identify PII entities in a legal text and return a structured JSON response.
>
> Return ONLY the JSON object described in the instructions. No explanation, no markdown fencing, no preamble.
>
> {filled_anonymize_template}

Where `{filled_anonymize_template}` is `prompts/anonymize.txt` with:
- `{existing_mapping}` replaced with `{}`
- `{message_text}` replaced with `"{prompt_prefix}\n\n{content}"` from the test case

**Parallelize**: spawn up to 4 Haiku agents at once for speed. Wait for batch to complete before next batch.

Parse each agent's response as JSON. If it returns markdown-fenced JSON, strip the fences. If parsing fails, mark as error and move on.

## Step 4: Apply mapping and check for leaks

For each successful anonymization result:

1. Extract the `entities_found` array from the JSON
2. Build a mapping: `{entity.text: entity.replacement}` for all entities with `action: "replace"` or `action: "generalize"`
3. Apply the mapping to the original text to produce sanitized text (replace each original with its placeholder)
4. Check for leaks: scan the sanitized text for any original PII values that weren't replaced

## Step 5: Evaluate (you are the judge)

For each test case, evaluate the anonymization on three criteria (0-10 each):

### PII Leak Detection (score 0-10)
Scan the sanitized text for any real PII that survived. Check: names, companies, emails, phones, addresses, SSNs, account numbers. Score 10 = no leaks. Score 0 = major PII in plaintext.

### Semantic Coherence (score 0-10)
Does the sanitized text read logically? Are placeholders consistent and descriptive? Can a reader understand document structure and legal relationships? Score 10 = perfectly readable. Score 0 = incoherent.

### Completeness (score 0-10)
Did the anonymizer catch ALL significant PII? Compare detected entities against what's visible in the original. Missing a name or email is worse than missing a generic title. Score 10 = every significant entity caught. Score 0 = most missed.

## Step 6: Report results

Print a summary table for each test case:

```
TEST: {name}
  Entities: {count} found, {replace_count} replaced, {keep_count} kept
  Leaks: {leak_list or "none"}
  Scores: PII={pii}/10  Coherence={coherence}/10  Completeness={completeness}/10  Overall={avg}/10
```

Then print aggregate summary:

```
SUMMARY: {passed}/{total} cases evaluated
  Average PII Leak Detection: {avg}/10
  Average Semantic Coherence: {avg}/10
  Average Completeness: {avg}/10
  Overall Average: {avg}/10
```

Flag any case scoring below 6 on any criterion as needing attention.
