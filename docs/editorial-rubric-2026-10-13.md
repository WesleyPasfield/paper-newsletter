# Safe agent deployment and loop engineering

Wesley's October 13 feedback puts application-layer agent governance and loop engineering at equal priority. They serve the same deployment cycle: evaluation and feedback accelerate improvements, controls govern deployment and agent actions, and production outcomes guide the next iteration. Provider safeguards are a secondary interest where they inform that cycle.

The rubric now gives this combined interest 60% editorial emphasis. Architecture receives 20%, production reliability 15%, and transferable enterprise engineering 5%. These percentages guide the prompts; the arithmetic score remains 40% direct fit, 25% evidence, 20% production transferability, 10% domain-expert relevance, and 5% novelty. The same top direct-fit range is available to governance and loop-engineering work independently.

The deployment focus includes authority and tool-access boundaries, runtime policies, containment, oversight, coordination between agents, and harmful effects across services. It covers unsafe behavior during benign tasks as well as attacks. Loop engineering includes runtime adaptation and evaluations of persistent improvement, including negative findings. Neither side needs a production deployment to earn high topical fit; evidence and transferability are separate assessments.

## Curation and source context

[X-AgentRed](https://arxiv.org/abs/2610.13886) is Wesley's explicit positive example and appears in the scoring prompt. Its feedback is recorded in `tests/content_eval_governance_examples.json` as positive relevance, without an invented exact rank or a must-include requirement. The other seven papers have null human labels. Source summaries in that file are paraphrases; live scoring used the primary arXiv abstracts instead.

The requested focus is consistent with the control failures described in the [Hugging Face incident report](https://openai.com/index/hugging-face-incident-and-the-road-ahead/), the [Australian government's account](https://ministers.finance.gov.au/financeminister/transcript/2026/09/24/press-conference-sydney), and [Australian guidance on agent harnesses](https://www.cyber.gov.au/advisory/agentic-ai-harnesses). These sources clarify the application-control scope. News and incident descriptions were not added to the paper-selection corpus or prompt.

## Development comparison

Compared the production rubric at `57668ca8f8a7c74710d66529018e45a595d5a96a` with the revised rubric using `gpt-5.6-luna`, low reasoning effort for abstract scoring, and two independent evaluations per paper per rubric. Only title, paper ID, and the primary abstract were passed as candidate content. Human labels and explanatory source summaries were excluded from the live inputs. No newsletter handler or sending function was invoked.

Mean scores across two runs:

| Paper | Previous total | Revised total | Previous direct fit | Revised direct fit |
| --- | ---: | ---: | ---: | ---: |
| X-AgentRed | 0.851 | 0.859 | 0.940 | 0.970 |
| RATED | 0.853 | 0.825 | 0.940 | 0.890 |
| FORTE | 0.846 | 0.631 | 0.870 | 0.520 |
| Adaptive DHiSparse | 0.782 | 0.663 | 0.820 | 0.625 |
| MC-EvoBench | 0.807 | 0.776 | 0.960 | 0.885 |
| Matched-Context Group Policy Optimization | 0.805 | 0.739 | 0.935 | 0.840 |

The revision places the cross-service safety study first in this small comparison while retaining the feedback-driven agent papers near the top. Adjacent recommender and inference engineering receive lower fit. The first draft over-penalized runtime adaptation, so the final rubric explicitly includes feedback-driven tool selection and escalation along with controlled evaluations of improvement.

Two additional probes were chosen after that revision and are not named in the prompt. Across two runs, AgentDojo was classified as `agent_governance` with mean direct fit 0.950 and total 0.867. Ask Your Advisor was classified as `loop_engineering` with mean direct fit 0.905 and total 0.783. These provide a small check that both topics can reach the top fit band beyond the named anchor. AgentDojo is historical test material and would fail the production freshness window.

This is a development check, not a held-out benchmark or a calibrated accuracy estimate. X-AgentRed is in the prompt, the other labels are not human judgments, and two samples provide little information about variance. Title screening and abstract ranking are separate stages: the table evaluates every abstract for comparability even where the title score would have fallen below the screen. It does not reconstruct an actual weekly issue. More human feedback on future selections is needed to assess ranking quality.

## Deployment traceability

Newly generated issues record `editorial_rubric_version=2026-10-13-safe-agent-deployment`. Existing archives and duplicate exclusions continue to apply. The curated examples do not bypass freshness, evidence assessment, or previous-publication checks.
