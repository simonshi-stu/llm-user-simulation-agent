# Third-Party Notices

This repository depends on the following external project:

## websocietysimulator

- Package: `websocietysimulator`
- Version used by this repository: `1.0.0a30` (pinned in `requirements.txt`)
- Upstream repository:
  [tsinghua-fib-lab/AgentSocietyChallenge](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge)
- Upstream license:
  [MIT](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge/blob/main/LICENSE)
- Copyright: Copyright (c) 2024 FIB LAB, Tsinghua University
- Original project: WWW'25 AgentSociety Challenge
- Purpose: simulation environment, simulator, task interaction tools, agent
  base classes and framework evaluation interfaces
- Original authors include Yuwei Yan and other AgentSociety contributors

The framework is installed from PyPI as a dependency and is not vendored or
copied into this repository. Its original license and attribution remain
applicable to the dependency. The framework's MIT license does not set the
license terms for Amazon, Yelp or Goodreads data, which are not distributed
here.

## Related project (not a direct dependency)

- [AgentSociety](https://github.com/tsinghua-fib-lab/AgentSociety) is the wider
  LLM social-simulation platform from the same lab. This repository does not
  vendor or install AgentSociety itself; the direct dependency above is the
  AgentSocietyChallenge package.
- AgentSociety is Apache-2.0 licensed. That license does not automatically
  apply to this repository or to the challenge package; each project keeps its
  own license and attribution.
