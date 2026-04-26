# Web Search Integration

This document explains how the Web Search agent integrates DuckDuckGo or Google search results, summarizes them using the local LLM, and returns structured output to the manager.

- Workflow: search → fetch → summarize by LLM
- Output: title, snippet, href, summary
