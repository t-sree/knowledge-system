"""The knowledge system behind `solution.KnowledgeSystem`.

  ingest/     read snapshots: page store, page classifier, fact extractor
  facts/      what is true when: fact history, resolver, timeline
  query/      understand the question: parser, vocabulary learned from the site
  answering/  build the answer: from facts, or from the site's own sentences
  llm/        OpenAI access and LLM question understanding (optional)
  conflicts   contradictions on the site
  pipeline    wires the parts together
  text        text helpers shared by all parts
"""
